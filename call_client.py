from azure.iot.hub import IoTHubRegistryManager
from azure.iot.hub.models import CloudToDeviceMethod
import os
import json
import time

def invoke_module_method(connection_string, device_id, module_id, method_name, method_payload):
    try:
        # Create IoTHubRegistryManager instance
        registry_manager = IoTHubRegistryManager(connection_string)

        # Build direct method request
        device_method = CloudToDeviceMethod(
            method_name=method_name,
            payload=method_payload,
            response_timeout_in_seconds=30,
            connect_timeout_in_seconds=30
        )

        # Invoke the direct method on the module
        response = registry_manager.invoke_device_module_method(
            device_id,
            module_id,
            device_method
        )

        print("Direct method invoked")
        print("Response status: {}".format(response.status))
        print("Response payload: {}".format(response.payload))

        return response

    except Exception as ex:
        print(ex)
        print("Error invoking method: {}".format(ex.with_traceback))
        return None

if __name__ == "__main__":
    # Replace these with your actual values
    CONNECTION_STRING = "HostName=dedee-devices.azure-devices.net;SharedAccessKeyName=iothubowner;SharedAccessKey=Gs8KVFhuiB2coy3NVJA6bGDW3QDBiy6v5AIoTEzUutM="
    DEVICE_ID = "dedee-edge-1"
    MODULE_ID = "ollama"
    METHOD_NAME = "generate"
    METHOD_PAYLOAD = {
        "model": "llama3",
        "prompt": "Who is the president of the USA?"
    }

    print("Invoke started ... ")
    # Invoke the method
    response = invoke_module_method(
        CONNECTION_STRING,
        DEVICE_ID,
        MODULE_ID,
        METHOD_NAME,
        METHOD_PAYLOAD
    )
    print("Invoke completed ... ")
    print(response)