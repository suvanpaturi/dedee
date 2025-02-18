import asyncio
from azure.iot.hub import IoTHubRegistryManager
from azure.iot.device import MethodResponse

# Replace with your IoT Hub connection string
CONNECTION_STRING = "HostName=dedee-devices.azure-devices.net;SharedAccessKeyName=iothubowner;SharedAccessKey=Gs8KVFhuiB2coy3NVJA6bGDW3QDBiy6v5AIoTEzUutM="

# Initialize the registry manager
registry_manager = IoTHubRegistryManager(CONNECTION_STRING)

device_id = "dedee-edge-1"
module_id = "llm_module"
method_name = "process_query"
payload = {"question": "What is the capital of France?"}
timeout = 30

response = registry_manager.invoke_device_module_method(device_id, module_id, method_name, payload, timeout)
print(response)