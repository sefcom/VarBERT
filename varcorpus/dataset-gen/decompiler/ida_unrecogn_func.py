import idaapi
import idc
import ida_funcs
import ida_hexrays
import ida_kernwin
import ida_loader

def setup():
    global analysis
    path = ida_loader.get_path(ida_loader.PATH_TYPE_CMD)

def read_list(filename):
    with open(filename, 'r') as r:
        ty_addrs = r.read().split('\n')
    return ty_addrs

def add_unrecognized_func(ty_addrs):  
    for addr in ty_addrs:
        # check if func is recognized by IDA
        name = ida_funcs.get_func_name(int(addr))
        if name:
            print(f"func present at: {addr} {name}")
        else:
            if ida_funcs.add_func(int(addr)):
                print(f"func recognized at: {addr} ")
            else:
                print(f"bad address {addr}")

def go():
    setup()
    ea = 0    
    filename = idc.ARGV[1]
    ty_addrs = read_list(filename)
    add_unrecognized_func(ty_addrs)       
    
    while True:
        func = ida_funcs.get_next_func(ea)
        if func is None:
            break
        ea = func.start_ea
        seg = idc.get_segm_name(ea)
        if seg != ".text":
            continue
        print('analyzing', ida_funcs.get_func_name(ea), hex(ea), ea)
        analyze_func(func)
    
def analyze_func(func):
    cfunc = ida_hexrays.decompile_func(func, None, 0)
    if cfunc is None:
        return

idaapi.auto_wait()
go()
