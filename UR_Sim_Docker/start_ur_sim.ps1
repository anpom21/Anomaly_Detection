# Start VcXsrv X server
Write-Host "Starting X server (VcXsrv)..."
Start-Process -FilePath "$env:ProgramFiles\VcXsrv\vcxsrv.exe" -ArgumentList ":0 -multiwindow -clipboard -wgl -ac"

# Wait a few seconds to ensure X server is running
Start-Sleep -Seconds 3

# Start Docker Desktop
Write-Host "Starting Docker Desktop..."
Start-Process -FilePath "C:\Program Files\Docker\Docker\Docker Desktop.exe"

# Wait for Docker to become responsive (optional: check docker status in loop)
Write-Host "Waiting for Docker to be ready..."
Start-Sleep -Seconds 10

# Run the URSim Docker container
Write-Host "Starting URSim container..."
docker run --rm `
    -e DISPLAY=host.docker.internal:0 `
    -p 30001-30004:30001-30004 `
    -p 29999:29999 `
    universalrobots/ursim_e-series

# If it doesn't work can you copy this into PS manualy:
#& "$env:ProgramFiles\VcXsrv\vcxsrv.exe" :0 -multiwindow -clipboard -wgl -ac # Runs
## X server
#docker desktop start; # Starts docker desktop
#docker run --rm `
#-e DISPLAY=host.docker.internal:0 `
#-p 30001-30004:30001-30004 `
#-p 29999:29999 `
#universalrobots/ursim_e-series # Runs UR Sim and opens ports for UR RTDE