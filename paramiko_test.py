import paramiko
import sys
import time
import getch  # Import the getch library

# Configura la conexión SSH
host = '192.168.80.3'
port = 20022
username = 'spot'
password = 'Merkleb0t'

# Crea una instancia SSHClient
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    ssh.connect(host, port, username, password)
    print("Conexión SSH establecida")

    print("Press keys (no need to press Enter) to send to the remote script.")
    print("Press 'q' to quit.")

    while True:
        char = getch.getch()  # Capture individual keypresses
        
        if char == 'q':
            break  # Quit if 'q' is pressed

        # Construye el comando para ejecutar el programa en el robot con el carácter como argumento
        comando = f'python3.6 -u /home/spot/julen_yash/julen.py {char}'

        print(f'CHAR: {char}')
        print('EXECUTED COMMAND:', comando)

        # Ejecuta el comando en the robot
        stdin, stdout, stderr = ssh.exec_command(comando)

        # Wait for the command to complete
        stdout.channel.recv_exit_status()

        # Print the standard output and error
        print("STDOUT:", stdout.read().decode())
        print("STDERR:", stderr.read().decode())

finally:
    # Cierra la conexión SSH
    ssh.close()
    print("Conexión SSH cerrada")
