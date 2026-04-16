import re
import time
def parse_qe_output(qe_output_file,dump_file):
    """ Parse Quantum ESPRESSO output file to extract atomic positions """
    timestep=0
    xlo= -0.0
    xhi= 10.53
    ylo= -0.0
    yhi= 10.53
    zlo= -0.0
    zhi= 10.53
    tatoms=54
    framestoc=1
    with open(dump_file, 'w') as g:
        with open(qe_output_file, 'r') as f:
            lines = f.readlines()
        linesinfile = len(lines)
        atomic_positions = []
        flagr=0
        i=0
        ftoc=0
        stacompa="ATOMIC_POSITIONS (angstrom)"
        for line in lines:
        # Example regular expression for extracting atomic positions from a typical QE output
        #print(line.strip())# control+c para cancelar
        #time.sleep(0.05)
            i=i+1
            if i==400000:
                print("going well")
                i=0
            if line.strip()=="":
                flagr=0
            if flagr==1:
                an=line.strip()
                ant=an.split()
                g.write(f'{ii} 1 {ant[1]} {ant[2]} {ant[3]}\n')
                ii=ii+1
            if line.strip()==stacompa:
                ftoc=ftoc+1
                if ftoc==framestoc:
                    flagr=1
                    ii=1
                    g.write(f'ITEM: TIMESTEP\n{timestep}\n')
                    g.write(f'ITEM: NUMBER OF ATOMS\n{tatoms}\n')
                    g.write(f'ITEM: BOX BOUNDS pp pp pp\n')
                    g.write(f'{xlo} {xhi}\n')
                    g.write(f'{ylo} {yhi}\n')
                    g.write(f'{zlo} {zhi}\n')
                    g.write(f'ITEM: ATOMS id type x y z\n')
                    timestep=timestep+1
                    ftoc=0
        print(f'OVITO dump file "{dump_file}" created successfully.')
        print("Total lines in the file is:",linesinfile)
        print("Last timestep is:",timestep)

if __name__ == "__main__":
    qe_output_file = 'BCC_54.out'
    dump_file = 'output.dump'

    parse_qe_output(qe_output_file,dump_file)

