from pypylon import pylon
import numpy as np
import cv2
import math 


# === Connect to the camera ===
def connect_camera():
    dc = pylon.DeviceInfo()
    dc.SetDeviceClass("BaslerGTC/Basler/GenTL_Producer_for_Basler_blaze_101_cameras")
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice(dc))
    camera.Open()
    camera.GenDCStreamingMode.Value = "Off"
    return camera

# === Get images (intensity and depth) ===
def get_images(grabResult):
    """
    Hämtar och returnerar djupkarta och intensitetsbild
    
    """

    # Data grabbed successfully?
    if grabResult.GrabSucceeded():
        # Get the grab result as a PylonDataContainer
        pylonDataContainer = grabResult.GetDataContainer()

        # Access data components if the component type indicates image data
        for componentIndex in range(pylonDataContainer.DataComponentCount):
            pylonDataComponent = pylonDataContainer.GetDataComponent(componentIndex)
            if pylonDataComponent.ComponentType == pylon.ComponentType_Intensity:
                intensity = pylonDataComponent.Array
                latest_intensity_image = intensity.reshape(pylonDataComponent.Height, pylonDataComponent.Width)
            elif pylonDataComponent.ComponentType == pylon.ComponentType_Range:
                # Point cloud data (depth) - no need to reshape to (h, w, 3)
                pointcloud = pylonDataComponent.Array
                latest_depth_image = pointcloud.reshape(pylonDataComponent.Height, pylonDataComponent.Width)
                
    return latest_intensity_image, latest_depth_image


# === Continuous image grabbing loop ===
def start_grabbing(camera):
    
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    while camera.IsGrabbing():
        grabResult = camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
        intensity, depth = get_images(grabResult)
        break

    grabResult.Release()
    camera.StopGrabbing()
    return intensity, depth

def get_3d_coordinates(center, depth_map, camera):

    
    """
    Hämtar djupbild från kameran och beräknar 3D-koordinat från en given center-punkt.

    Args:
        center (tuple): (x_pixel, y_pixel) i bilden
    
    Returns:
        (x_mm, y_mm, z_mm): position i millimeter relativt kamerafronten
    """

    # === Kamera-parametrar ===

    gray2mm = camera.Scan3dCoordinateScale.GetValue()
    f = camera.Scan3dFocalLength.GetValue()

    # Hämtar kamera inställningar för att få rätt värden
    cx = camera.Scan3dPrincipalPointU.GetValue()
    cy = camera.Scan3dPrincipalPointV.GetValue()
    z_offset = camera.ZOffsetOriginToCameraFront.GetValue()

    # === Runda center-pixel till int
    u = int(round(center[0]))
    v = int(round(center[1]))

    # === Läs depth value
    g = depth_map[v, u]

    if g == 0:
        print(f" Ingen giltig depth på ({u}, {v})!")
        return None, None, None

    # === Beräkna z i mm
    z = g * gray2mm
    z_corrected = z - z_offset  # korrigerad Z från kamerafront

    # === Beräkna x och y i mm
    x = (u - cx) * z_corrected/ f
    y = (v - cy) * z_corrected / f

    return x, y, z_corrected

# === Stoppa kamera ===
def stop_grabbing():
    global stop_camera
    stop_camera = True

# --- Example usage ---
if __name__ == "__main__":
    start_grabbing(camera=connect_camera())

   