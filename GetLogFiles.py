import paramiko
import os

def create_ssh_client(remote_ip, remote_port, remote_user, remote_password):
    # Create SSH client
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        # Connect to the remote server
        ssh.connect(remote_ip, port=remote_port, username=remote_user, password=remote_password)
        return ssh
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def close_ssh_client(ssh):
    ssh.close()

def create_sftp_client(ssh):
    return paramiko.SFTPClient.from_transport(ssh.get_transport())

def close_sftp_client(sftp):
    sftp.close()

def get_list_of_files(sftp, remote_path):
    return sftp.listdir(remote_path)

def copy_files_by_date(sftp, date , remote_path='/Documents', local_path='C:\\Users\\user\\Data'):
    remote_ip_address = sftp.get_channel().get_transport().getpeername()[0]
    remote_path = f'{remote_path}/{date}'
    local_path = f'{local_path}\\{remote_ip_address}\\{date}'
    time_folders = get_list_of_files(sftp, remote_path)
    for folder in time_folders:
        ff = get_list_of_files(sftp, f'{remote_path}/{folder}')
        for f in ff:
            if f == 'Detections':
                detections = get_list_of_files(sftp, f'{remote_path}/{folder}/{f}')
                for d in detections:
                    print(d)
                    if not os.path.exists(f'{local_path}\\{folder}\\{f}'):
                        os.makedirs(f'{local_path}\\{folder}\\{f}')
                        os.chmod(f'{local_path}\\{folder}\\{f}', 0o777)
                        
                        
                    sftp.get(f'{remote_path}/{folder}/{f}/{d}', f'{local_path}\\{folder}\\{f}\\{d}')

# Example usage
remote_ip = '2.54.88.254'
remote_user = 'g188'
remote_password = '1470'
remote_port = 51807


ssh = create_ssh_client(remote_ip, remote_port, remote_user, remote_password)
sftp = create_sftp_client(ssh)

date = '12-03-2025'
copy_files_by_date(sftp, date)
