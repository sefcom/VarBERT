import os
import time
import shutil
import logging
import subprocess
from pathlib import Path

l = logging.getLogger('main')

def decompile_(decompiler_workdir, decompiler_path, binary_name, binary_path, binary_type, 
               decompiled_binary_code_path, failed_path , type_strip_addrs, type_strip_mangled_names):
    try:
        shutil.copy(binary_path, decompiler_workdir)
        current_script_dir = Path(__file__).resolve().parent
        addr_file = os.path.join(decompiler_workdir, f'{binary_name}_addr')
        if binary_type == 'strip':
            run_str = f'{decompiler_path} -S"{current_script_dir}/ida_unrecogn_func.py {addr_file}" -L{decompiler_workdir}/log_{binary_name}.txt -Ohexrays:{decompiler_workdir}/outfile:ALL -A {decompiler_workdir}/{binary_name}'
        elif binary_type == 'type_strip':
            run_str = f'{decompiler_path} -S"{current_script_dir}/ida_analysis.py {addr_file}" -L{decompiler_workdir}/log_{binary_name}.txt -Ohexrays:{decompiler_workdir}/outfile:ALL -A {decompiler_workdir}/{binary_name}'
        
        for _ in range(5):  
            subprocess.run([run_str], shell=True)
            if f'outfile.c' in os.listdir(decompiler_workdir):
                l.debug(f"outfile.c generated! for {binary_name}")
                break
            time.sleep(5)

    except Exception as e:
        shutil.move(os.path.join(decompiler_workdir, f'log_{binary_name}.txt'), os.path.join(failed_path, ('ida_' + binary_name + '.log')))
        return None, str(os.path.join(failed_path, ('ida_' + binary_name + '.log')))

    finally:
        # cleanup!
        if os.path.exists(os.path.join(decompiler_workdir, f'{binary_name}.i64')):
            os.remove(os.path.join(decompiler_workdir, f'{binary_name}.i64'))
            if binary_type == 'strip': 
                subprocess.run(['mv', addr_file, type_strip_addrs ])
                subprocess.run(['mv', f'{addr_file}_names', type_strip_mangled_names])     
        if os.path.exists(f'{decompiler_workdir}/outfile.c'):
            if subprocess.run(['mv', f'{decompiler_workdir}/outfile.c', f'{decompiled_binary_code_path}']).returncode == 0:
                return True, f'{decompiled_binary_code_path}'
        else:
            return False, str(os.path.join(failed_path, ('ida_' + binary_name + '.log')))