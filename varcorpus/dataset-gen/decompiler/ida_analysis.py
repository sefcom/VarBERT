import idaapi
import idc
import ida_funcs
import ida_hexrays
import ida_kernwin
import ida_loader

from collections import namedtuple, defaultdict
from sortedcontainers import SortedDict

from elftools.dwarf.descriptions import describe_reg_name
from elftools.elf.elffile import ELFFile
from elftools.dwarf.dwarf_expr import DWARFExprParser
from elftools.dwarf import locationlists

import json

LocationEntry = namedtuple("LocationEntry", ("begin_offset", "end_offset", "location"))
NameResult = namedtuple("NameResult", ("name", "size"))

class RegVarAnalysis:
    def __init__(self, fname):
        a = ELFFile(open(fname, 'rb'))
        b = a.get_dwarf_info()

        self.result = defaultdict(SortedDict)

        self.loc_parser = b.location_lists()
        self.expr_parser = DWARFExprParser(b.structs)
        self.range_lists = b.range_lists()

        for c in b.iter_CUs():
            for x in c.iter_DIEs():
                if x.tag != 'DW_TAG_variable' or 'DW_AT_name' not in x.attributes or x.get_parent() is x.cu.get_top_DIE() or 'DW_AT_location' not in x.attributes:
                    continue
                # ???
                if x.attributes['DW_AT_location'].form == 'DW_FORM_exprloc':
                    loclist = self.get_single_loc(x)
                elif x.attributes['DW_AT_location'].form != 'DW_FORM_sec_offset':
                    assert False
                else:
                    loclist = self.get_loclist(x)
                for loc in loclist:
                    if len(loc.location) != 1 or not loc.location[0].op_name.startswith('DW_OP_reg'):
                        # discard complicated variables
                        continue
                    expr = loc.location[0]
                    if expr.op_name == 'DW_OP_regx':
                        reg_name = describe_reg_name(expr.args[0], a.get_machine_arch())
                    else:
                        reg_name = describe_reg_name(int(expr.op_name[9:]), a.get_machine_arch())
                    self.result[reg_name][loc.begin_offset] = NameResult(x.attributes['DW_AT_name'].value.decode(), loc.end_offset - loc.begin_offset)

    def get_single_loc(self, f):
        base_addr = 0
        low_pc = f.cu.get_top_DIE().attributes.get("DW_AT_low_pc", None)
        if low_pc is not None:
            base_addr = low_pc.value
        parent = f.get_parent()
        ranges = []
        while parent is not f.cu.get_top_DIE():
            if 'DW_AT_low_pc' in parent.attributes and 'DW_AT_high_pc' in parent.attributes:
                ranges.append((
                    parent.attributes['DW_AT_low_pc'].value + base_addr,
                    parent.attributes['DW_AT_high_pc'].value + base_addr,
                ))
                break
            if 'DW_AT_ranges' in parent.attributes:
                rlist = self.range_lists.get_range_list_at_offset(parent.attributes['DW_AT_ranges'].value)
                ranges = [
                    (rentry.begin_offset + base_addr, rentry.end_offset + base_addr) for rentry in rlist
                ]
                break
            parent = parent.get_parent()
        else:
            return []

        return [LocationEntry(
            location=self.expr_parser.parse_expr(f.attributes['DW_AT_location'].value),
            begin_offset=begin,
            end_offset=end
        ) for begin, end in ranges]


    def get_loclist(self, f):
        base_addr = 0
        low_pc = f.cu.get_top_DIE().attributes.get("DW_AT_low_pc", None)
        if low_pc is not None:
            base_addr = low_pc.value

        loc_list = self.loc_parser.get_location_list_at_offset(f.attributes['DW_AT_location'].value)
        result = []
        for item in loc_list:
            if type(item) is locationlists.LocationEntry:
                try:
                    result.append(LocationEntry(
                        base_addr + item.begin_offset,
                        base_addr + item.end_offset,
                        self.expr_parser.parse_expr(item.loc_expr)))
                except KeyError as e:
                    if e.args[0] == 249:  # gnu extension dwarf expr ops
                        continue
                    else:
                        raise
            elif type(item) is locationlists.BaseAddressEntry:
                base_addr = item.base_address
            else:
                raise TypeError("What kind of loclist entry is this?")
        return result

    def lookup(self, reg, addr):
        try:
            key = next(self.result[reg].irange(maximum = addr, reverse=True))
        except StopIteration:
            return None
        else:
            val = self.result[reg][key]
            if key + val.size <= addr:
                return None
            return val.name

analysis: RegVarAnalysis = None

def setup():
    global analysis
    if analysis is not None:
        return
    path = ida_loader.get_path(ida_loader.PATH_TYPE_CMD)
    analysis = RegVarAnalysis(path)

def dump_list(list_, filename):
    with open(filename, 'w') as w:
        w.write("\n".join(list_))

def write_json(data, filename):
    with open(filename, 'w') as w:
        w.write(json.dumps(data))

def go():
    setup()

    ea = 0
    collect_addrs, mangled_names_to_demangled_names = [], {}
    filename = idc.ARGV[1]
    while True:
        func = ida_funcs.get_next_func(ea)
        if func is None:
            break
        ea = func.start_ea
        seg = idc.get_segm_name(ea)
        if seg != ".text":
            continue
        collect_addrs.append(str(ea))
        typ = idc.get_type(ea)
        # void sometimes introduce extra variables, updating return type helps in variable matching for type-strip binary
        if 'void' in str(typ):
            newtype = str(typ).replace("void", f"__int64 {str(ida_funcs.get_func_name(ea))}") + ";"
            res = idc.SetType(ea, newtype)

        print("analyzing" , ida_funcs.get_func_name(ea))        
        analyze_func(func)
        # # Demangle the name
        mangled_name = ida_funcs.get_func_name(ea)
        demangled_name = idc.demangle_name(mangled_name, idc.get_inf_attr(idc.INF_SHORT_DN))
        if demangled_name:
            mangled_names_to_demangled_names[mangled_name] = demangled_name
        else:
            mangled_names_to_demangled_names[mangled_name] = mangled_name

    dump_list(collect_addrs, filename)
    write_json(mangled_names_to_demangled_names, f'{filename}_names')

def analyze_func(func):
    cfunc = ida_hexrays.decompile_func(func, None, 0)
    if cfunc is None:
        return
    v = Visitor(func.start_ea, cfunc)
    v.apply_to(cfunc.body, None)
    return v

class Visitor(idaapi.ctree_visitor_t):
    def __init__(self, ea, cfunc):
        super().__init__(idaapi.CV_FAST)
        self.ea = ea
        self.cfunc = cfunc
        self.vars = []
        self.already_used = {lvar.name for lvar in cfunc.lvars if lvar.has_user_name}
        self.already_fixed = set(self.already_used)

    def visit_expr(self, expr):
        if expr.op == ida_hexrays.cot_var:
            lvar = expr.get_v().getv()
            old_name = lvar.name
            if expr.ea == idc.BADADDR:
                pass
            else:
                if old_name not in self.already_fixed:
                    if lvar.location.is_reg1():
                        reg_name = ida_hexrays.print_vdloc(lvar.location, 8)
                        if reg_name in analysis.result:
                            var_name = analysis.lookup(reg_name, expr.ea)
                            if var_name:
                                nonce_int = 0
                                nonce = ''
                                while var_name + nonce in self.already_used:
                                    nonce = '_' + str(nonce_int)
                                    nonce_int += 1
                                name = var_name + nonce
                                ida_hexrays.rename_lvar(self.ea, old_name, name)
                                self.already_used.add(name)
                                self.already_fixed.add(old_name)

        return 0

idaapi.auto_wait()
go()
