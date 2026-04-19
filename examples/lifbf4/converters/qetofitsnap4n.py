import re
import time
from pathlib import Path
def parse_qe_output(qe_output_file,dump_file):
    
    def impresion(tatoms,energy,stress_matrix,positions_matrix,lattice_matrix,forces_matrix,types_matrix,numerofile):
        real_output=f'{dump_file}_{numerofile}.json'
        print(real_output)
        with open(real_output, 'w') as g:
            g.write(f'# A test JSON file for QM data\n')
            sttp = "{\"Dataset\": {\"Data\": [{\"Stress\": "
            g.write(f'{sttp}{stress_matrix}, ')
            sttp = "\"Positions\": "
            g.write(f'{sttp}{positions_matrix}, ')
            sttp = "\"Energy\": "
            g.write(f'{sttp}{energy}, ')
            sttp = "\"AtomTypes\": "
            g.write(f'{sttp}{types_matrix}, ')
            sttp = "\"Lattice\": "
            g.write(f'{sttp}{lattice_matrix}, ')
            sttp = "\"NumAtoms\": "
            g.write(f'{sttp}{tatoms}, ')
            sttp = "\"Forces\": "
            g.write(f'{sttp}{forces_matrix}')
            sttp = "}]"
            g.write(f'{sttp}, ')
            sttp = "\"PositionsStyle\": \"angstrom\", \"AtomTypeStyle\": \"chemicalsymbol\", "
            g.write(f'{sttp}')
            sttp = "\"Label\": \"Example containing 1 configurations, each with "+str(tatoms)+" atoms\", "
            g.write(f'{sttp}')
            sttp = "\"StressStyle\": \"bar\", \"LatticeStyle\": \"angstrom\", "
            g.write(f'{sttp}')
            sttp = "\"EnergyStyle\": \"electronvolt\", \"ForcesStyle\": \"electronvoltperangstrom\"}}"
            g.write(f'{sttp}')

    numerofile = 0
    noener=0.0
    limitedefiles = 5999
    temp = 0
    cont = 0
    flargtr = 0
    printflag=[0,0,0,0]
    xdis = 4.73333213 #dimension x
    ydis = 4.10115034 #dimension y
    zdis = 10.83966193 #dimension z
    energy = 0.0
    framestoc = 2 #frames between samples
    tatoms = 18 #number of atoms
    stress_matrix = [[0.0] * 3 for sm in range(3)]
    positions_matrix = [[0.0] * 3 for pm in range(tatoms)]
    forces_matrix = [[0.0] * 3 for fm in range(tatoms)]
    lattice_matrix = [[xdis, 0.0, 0.0], [2.36666607, ydis, 0.0], [0.0, 0.0, zdis]]
    types_matrix = []
    with open(qe_output_file, 'r') as f:
        lines = f.readlines()
        linesinfile = len(lines)
        icl=0
        contadordeline=0
        #Lines and Flags to determine matrixes (order for QE only)
        linetoforce="Forces acting on atoms (cartesian axes, Ry/au):"
        flagtf=0
        linetostress="total   stress  (Ry/bohr**3)"
        flagts=0
        linetopositions="ATOMIC_POSITIONS (angstrom)"
        flagtp=0
        linetoenergy="kinetic energy (Ekin)"
        flagte=0
        
        for line in lines:
        # Example regular expression for extracting atomic positions from a typical QE output
        #print(line.strip())# control+c para cancelar
        #time.sleep(0.05)
            icl=icl+1
            if numerofile < limitedefiles and printflag==[1,1,1,1]:
                printflag = [0,0,0,0]
                elements_with_double_quotes = [f'"{elem}"' for elem in types_matrix]
                types_matrix = "[" + ", ".join(elements_with_double_quotes) + "]"
                cont = cont + 1
                if cont >= framestoc:
                    impresion(tatoms,energy,stress_matrix,positions_matrix,lattice_matrix,forces_matrix,types_matrix,numerofile)
                    numerofile = numerofile + 1
                    cont = 0
                types_matrix = []
            if linetoforce in line:
                flagtf = 1
            if linetostress in line:
                flagts = 1
            if linetopositions in line:
                flagtp = 1
            if linetoenergy in line:
                flagte = 1
            if flagtf == 1:
                if temp >= 2:
                    an = line.strip()
                    ant = an.split()
                    forces_matrix[(temp-2)][0] = float(ant[6])*25.711
                    forces_matrix[(temp-2)][1] = float(ant[7])*25.711
                    forces_matrix[(temp-2)][2] = float(ant[8])*25.711
                temp = temp + 1
                if temp >= 2 + tatoms:
                    temp = 0
                    flagtf =0
                    printflag[0] = 1
            if flagts == 1:
                if temp >= 1:
                    an = line.strip()
                    ant = an.split()
                    stress_matrix[(temp-3)][0] = float(ant[3])*1000#from kbar to bar
                    stress_matrix[(temp-3)][1] = float(ant[4])*1000
                    stress_matrix[(temp-3)][2] = float(ant[5])*1000
                temp = temp + 1
                if temp >= 4:
                    temp = 0
                    flagts = 0
                    printflag[1] = 1
            if flagtp == 1:
                if temp >= 1:
                    an = line.strip()
                    ant = an.split()
                    positions_matrix[(temp-1)][0] = float(ant[1])
                    positions_matrix[(temp-1)][1] = float(ant[2])
                    positions_matrix[(temp-1)][2] = float(ant[3])
                    types_matrix.append(str(ant[0]))
                temp = temp + 1
                if temp >= 1 + tatoms:
                    temp = 0
                    flagtp = 0
                    printflag[2] = 1
            if flagte == 1:
                if temp == 0:
                    an = line.strip()
                    ant = an.split()
                    energy = float(ant[4])*13.605693#from Ry to eV
                temp = temp + 1
                if temp >= 3:
                    flagte = 0
                    an = line.strip()
                    ant = an.split()
                    energy = float(ant[5])*13.605693 - energy + noener*tatoms #from Ry to eV
                    printflag[3] = 1
                    temp = 0
        print("JSON file for fitsnap created successfully")
        print("Total lines in the file is:",linesinfile)
        print("Number of frames make is: ",numerofile )
if __name__ == "__main__":
    _example_dir = Path(__file__).resolve().parent.parent
    qe_output_file = str(_example_dir / "qe" / "LiBF4.out")
    fitsnap_file = str(_example_dir / "NEWJSON" / "output")

    parse_qe_output(qe_output_file, fitsnap_file)

