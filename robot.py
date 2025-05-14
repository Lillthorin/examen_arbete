import socket

"""
* Dataklass för att ansluta till robot och skicka och ta emot data 
* via socket (tcp/ip)
* Vid initiering anropas robotens ip och socket
"""

class PersistentRobotClient:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.ip, self.port))
    
    def send_coordinates(self, str_coords):
        print(f'sending coords: {str_coords}')
        target = str_coords
        self.sock.sendall(target.encode('utf-8'))
        return
    def receive_response(self):
        try:
            data = self.sock.recv(4096)
            return data.decode(errors='ignore').strip()
        except socket.timeout:
            return None

    def open_gripper(self):
        self.send_command("OPEN")

    def close_gripper(self):
        self.send_command("CLOSE")
    def close(self):
        self.sock.close()