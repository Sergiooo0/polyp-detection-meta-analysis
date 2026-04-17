import os
import socket
from fabric import Connection

def ping(host, port):
    print(f"Verifying connection to {host}:{port}...")
    try:
        socket.create_connection((host, port), timeout=2)
        return True
    except:
        return False

def transfer_folder(host, port, username, password, local_dir, remote_dir, tar_name):
    print(f"Zip the directory '{local_dir}'...")
    # Zip data directory into a tar.gz file
    os.system(f'tar -czf {tar_name} -C {local_dir} .')

    kwargs = {
        "password": str(password),
        "timeout" : 10
    }

    print(f"Connecting to ({username}@{host}:{port})...")
    with Connection(host=host, user=username, port=port, connect_kwargs=kwargs) as c:
        
        print(f"Creating remote directory '{remote_dir}' if it doesn't exist...")
        c.run(f'mkdir -p {remote_dir}')

        print("Transfering the compressed file to the Jetson...")
        c.put(tar_name, remote=f'/tmp/{tar_name}')

        print("Unzipping the file on the host...")
        c.run(f'tar -xzf /tmp/{tar_name} -C {remote_dir}')

        print("Cleaning up the compressed file on the host...")
        c.run(f'rm /tmp/{tar_name}')

    print("Cleaning up the local compressed file...")
    os.remove(tar_name)
    print("Transfer complete!")
