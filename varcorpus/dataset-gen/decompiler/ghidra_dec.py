import re
import os
import shutil
import logging
from ghidra.app.decompiler import DecompInterface, DecompileOptions

def decompile_(binary_name):

    binary = getCurrentProgram()
    decomp_interface = DecompInterface()
    options = DecompileOptions()
    options.setNoCastPrint(True)
    decomp_interface.setOptions(options)
    decomp_interface.openProgram(binary)

    func_mg = binary.getFunctionManager()
    funcs = func_mg.getFunctions(True) 

    # args 
    args = getScriptArgs()
    out_path, failed_path, log_path = str(args[0]), str(args[1]), str(args[2])  
 
    regex_com = r"(/\*[\s\S]*?\*\/)|(//.*)"
    tot_lines = 0
    
    try:
        with open(out_path, 'w') as w:
            for func in funcs:                        
                results = decomp_interface.decompileFunction(func, 0, None )    
                addr = str(func.getEntryPoint())
                func_res = results.getDecompiledFunction()
                if 'EXTERNAL' in func.getName(True):
                    continue
                if func_res:
                    func_c = str(func_res.getC())
                    # rm comments  
                    new_func = str(re.sub(regex_com, '', func_c, 0, re.MULTILINE)).strip()

                    # for joern parsing
                    before = '//----- ('
                    after = ') ----------------------------------------------------\n'
                    addr_line = before + addr + after
                    tot_lines += 2
                    w.write(addr_line)
                    w.write(new_func)
                    w.write('\n')
                    tot_lines += new_func.count('\n') 

    except Exception as e:
        # move log file to failed path
        shutil.move(os.path.join(log_path, (binary_name + '.log')), os.path.join(failed_path, ('ghidra_' + binary_name + '.log')))
 
if __name__ == '__main__':
    
    bin_name = str(locals()['currentProgram']).split(' - .')[0].strip()
    decompile_(bin_name)
    
