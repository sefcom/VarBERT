import sys
import random
import string
import logging
from elftools.elf.elffile import ELFFile
from elftools.dwarf import constants
import dwarfwrite
from dwarfwrite.restructure import ReStructurer, VOID
from cle.backends.elf.elf import ELF

l = logging.getLogger('main')
DWARF_VERSION = 4
VOID_STAR_ID = 127

class TypeStubRewriter(ReStructurer):
    def __init__(self, fp, cu_data=None):
        super().__init__(fp)
        fp.seek(0)
        self.rng = random.Random(fp.read())
        self.cu_data = cu_data

    def unit_get_functions(self, unit):
        result = list(super().unit_get_functions(unit))

        for die in unit.iter_DIEs():
            if die.tag not in ('DW_TAG_class_type', 'DW_TAG_struct_type'):
                continue
            for member in die.iter_children():
                if member.tag != 'DW_TAG_subprogram':
                    continue
                result.append(member)

        return result

    def type_basic_encoding(self, handler):
        return constants.DW_ATE_unsigned

    def type_basic_name(self, handler):
        return {
            1: "unsigned char",
            2: "unsigned short",
            4: "unsigned int",
            8: "unsigned long long",
            16: "unsigned __int128",
            32: "unsigned __int256",
        }[handler]

    def function_get_linkage_name(self, handler):
        return None

    def parameter_get_name(self, handler):
        return super().parameter_get_name(handler)
    

def get_type_size(cu, offset, wordsize):
    # returns type_size, base_type_offset, is_exemplar[true=yes, false=maybe, none=no]
    type_die = cu.get_DIE_from_refaddr(offset + cu.cu_offset)
    base_type_die = type_die
    base_type_offset = offset

    while True:
        if base_type_die.tag == 'DW_TAG_pointer_type':
            return wordsize, base_type_offset, None
        elif base_type_die.tag == 'DW_TAG_structure_type':
            try:
                first_member = next(base_type_die.iter_children())
                assert first_member.tag == 'DW_TAG_member'
                assert first_member.attributes['DW_AT_data_member_location'].value == 0
            except (KeyError, StopIteration, AssertionError):
                return wordsize, base_type_offset, None
            base_type_offset = first_member.attributes['DW_AT_type'].value
            base_type_die = cu.get_DIE_from_refaddr(base_type_offset + cu.cu_offset)
            continue
        # TODO unions
        elif base_type_die.tag == 'DW_TAG_base_type':
            size = base_type_die.attributes['DW_AT_byte_size'].value
            if any(ty in base_type_die.attributes['DW_AT_name'].value for ty in (b'char', b'int', b'long')):
                return size, base_type_offset, True
            else:
                return size, base_type_offset, None
        elif 'DW_AT_type' in base_type_die.attributes:
            base_type_offset = base_type_die.attributes['DW_AT_type'].value
            base_type_die = cu.get_DIE_from_refaddr(base_type_offset + cu.cu_offset)
            continue
        else:
            # afaik this case is only reached for void types
            return wordsize, base_type_offset, None

def build_type_to_size(ipath):
    with open(ipath, 'rb') as ifp:
        elf = ELFFile(ifp)
        arch = ELF.extract_arch(elf)

        ifp.seek(0)
        dwarf = elf.get_dwarf_info()

        cu_data = {}
        for cu in dwarf.iter_CUs():
            type_to_size = {}
            # import ipdb; ipdb.set_trace()
            cu_data[cu.cu_offset] = type_to_size
            for die in cu.iter_DIEs():
                attr = die.attributes.get("DW_AT_type", None)
                if attr is None:
                    continue

                type_size, _, _ = get_type_size(cu, attr.value, arch.bytes)
                type_to_size[attr.value] = type_size

    return cu_data

class IdaStubRewriter(TypeStubRewriter):

    def get_attribute(self, die, name):
        r = super().get_attribute(die, name)
        if name != 'DW_AT_type' or r is None:
            return r
        size = self.cu_data[r.cu.cu_offset][r.offset - r.cu.cu_offset]
        return size

    def type_basic_size(self, handler):
        return handler

    def function_get_name(self, handler):
        r = super().function_get_name(handler)
        return r

    def parameter_get_location(self, handler):
        return None

class GhidraStubRewriter(TypeStubRewriter):

    def __init__(self, fp, cu_data=None, low_pc_to_funcname=None):
        super().__init__(fp, cu_data)
        self.low_pc_to_funcname = low_pc_to_funcname

    def get_attribute(self, die, name):
        r = super().get_attribute(die, name)
        if name != 'DW_AT_type' or r is None:
            return r
        if die.tag == "DW_TAG_formal_parameter":
            varname = self.get_attribute(die, "DW_AT_name")
            if varname == b"this":
                return VOID_STAR_ID
        size = self.cu_data[r.cu.cu_offset][r.offset - r.cu.cu_offset]
        return size

    def type_basic_size(self, handler):
        if handler == VOID_STAR_ID:
            return 8  # assuming the binary is 64-bit
        return handler

    def function_get_name(self, handler):
        # import ipdb; ipdb.set_trace()
        low_pc = self.get_attribute(handler, "DW_AT_low_pc")
        if low_pc is not None and str(low_pc) in self.low_pc_to_funcname:
            return self.low_pc_to_funcname[str(low_pc)]
        return super().function_get_name(handler)

    def type_ptr_of(self, handler):
        if handler == VOID_STAR_ID:
            return VOID
        return None
    
    def parameter_get_artificial(self, handler):
        return None

    def parameter_get_location(self, handler):
        return super().parameter_get_location(handler)


def get_spec_offsets_and_names(ipath):

    elf = ELFFile(open(ipath, "rb"))
    all_spec_offsets = [ ]
    low_pc_to_funcname = {}
    spec_offset_to_low_pc = {}

    dwarf = elf.get_dwarf_info()
    for cu in dwarf.iter_CUs():
        cu_offset = cu.cu_offset
        for die in cu.iter_DIEs():
            for subdie in cu.iter_DIE_children(die):
                if subdie.tag == "DW_TAG_subprogram":
                    if "DW_AT_low_pc" in subdie.attributes:
                        low_pc = str(subdie.attributes.get("DW_AT_low_pc").value)
                        if "DW_AT_specification" in subdie.attributes:
                            spec_offset = subdie.attributes["DW_AT_specification"].value
                            global_spec_offset = spec_offset + cu_offset
                            all_spec_offsets.append(global_spec_offset)
                            spec_offset_to_low_pc[str(global_spec_offset)] = low_pc

    for cu in dwarf.iter_CUs():
        for die in cu.iter_DIEs():
            for subdie in cu.iter_DIE_children(die):
                if subdie.offset in all_spec_offsets:
                    if "DW_AT_name" in subdie.attributes:
                        low_pc_to_funcname[spec_offset_to_low_pc[str(subdie.offset)]] = subdie.attributes.get("DW_AT_name").value

    return low_pc_to_funcname

def type_strip_target_binary(ipath, opath, decompiler):
    try:
        cu_data = build_type_to_size(ipath)
        if decompiler == "ghidra":
            low_pc_to_funcname = get_spec_offsets_and_names(ipath)
            GhidraStubRewriter.rewrite_dwarf(in_path=ipath, out_path=opath, cu_data=cu_data, low_pc_to_funcname=low_pc_to_funcname)
        elif decompiler == "ida":
            IdaStubRewriter.rewrite_dwarf(in_path=ipath, out_path=opath, cu_data=cu_data)
        else:
            l.error("Unsupported Decompiler. Please choose from IDA or Ghidra")
    except Exception as e:
        l.error(f"Error occured while creating a type-strip binary: {e}")
