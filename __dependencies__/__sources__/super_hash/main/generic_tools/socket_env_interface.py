import socket
import atexit

def _send_message(obj, connection):
    message_content = obj
    size = len(message_content)
    size = size.to_bytes(INT_BYTE_LIMIT, 'big')
    connection.sendall(size)
    connection.sendall(message_content)

def _receive_message(obj, connection):
    data = connection.recv(INT_BYTE_LIMIT)
    if len(data) == 0:
        return None
    
    size = int.from_bytes(data, "big")
    data = connection.recv(size)
    return data
    
def setup_connection(*, server_ip, port, observation_to_bytes=lambda each: each, bytes_to_observation=lambda each: each, bytes_to_action=lambda each: each, action_to_bytes=lambda each: each):
    """
        Example:
            import numpy as np
            import cv2

            def observation_to_bytes(observation):
                return np.array(observation).tobytes()
                
            def bytes_to_observation(data):
                image = np.frombuffer(data, dtype=np.uint8)
                image = cv2.imdecode(image, cv2.COLOR_BGR2GRAY)
                return image

            def action_to_bytes(action):
                return bytes(json.dumps(action))
                
            def bytes_to_action(data):
                json.loads(data)
                return image

            AgentConnecter, EnvironmentConnector = setup_connection(
                server_ip=server_ip,
                port=port,
                observation_to_bytes=observation_to_bytes,
                bytes_to_observation=bytes_to_observation,
                action_to_bytes=action_to_bytes,
                bytes_to_action=bytes_to_action,
            )
            
            # on the Agent side
            connection = AgentConnecter()
            connection.receive_observation()
            connection.send_action()
            
            # on the Environment side
            connection = EnvironmentConnector()
            connection.send_observation()
            connection.receive_action()
    """
    class AgentConnecter:
        def __init__(self):
            # Connect to car
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.bind((server_ip, port))
            self.socket.listen()
            self.connection, self.address = self.socket.accept()
            atexit.register(self.connection.close)

        def receive_observation(self):
            return bytes_to_observation(
                _receive_message(self.connection)
            )

        def send_action(self, action):
            return _send_message(
                action_to_bytes(action),
                self.connection
            )
                
    class EnvironmentConnector:
        def __init__(self, *):
            # Connect to car
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((server_ip, port))
            atexit.register(self.socket.close)
            
        def send_observation(self, observation):
            return _send_message(
                observation_to_bytes(observation),
                self.connection,
            )

        def receive_action(self):
            return bytes_to_action(
                _receive_message(self.connection)
            )
    
    return AgentConnecter, EnvironmentConnector