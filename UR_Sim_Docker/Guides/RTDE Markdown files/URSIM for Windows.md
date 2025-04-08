# Installing URSim on Windows 10

Follow these steps to get URSim running via Docker on Windows 10 probably also works for Windows 11 but I haven't tested it.

## 1. Install Docker Desktop

- **Download and Install:**  
  Visit the [Docker Desktop download page](https://www.docker.com/products/docker-desktop) and install it.
  
- **Prerequisites:**  
  - Ensure virtualization is enabled in your BIOS.
  - If you're on Windows 10 Home, set up WSL2 as Docker Desktop requires it.
  - Confirm that Docker is set up to use Linux containers (default setup).

## 2. Prepare an X Server for the GUI

URSim has a GUI, and Docker containers on Windows don’t natively display GUIs. You’ll need an X server to forward the display.

- **Recommended X Server:**  
  [VcXsrv download link](https://sourceforge.net/projects/vcxsrv/) 

- **Install and Configure:**  
  After installing VcXsrv, launch it and allow network connections so that the container can forward its GUI to your Windows desktop.

## 3. Pull the URSim Image

Open PowerShell or Command Prompt and run:

```bash
docker pull universalrobots/ursim_e-series
```

This command downloads the URSim simulator image.

## 4. Start Docker Desktop
Run ```docker desktop start``` in powershell.
```bash
docker desktop start
```

## 5. Start docker dekstop and run the URSim Container with GUI Forwarding

Since Windows doesn’t support Linux’s `--net=host`, you need to manually map the display. Assuming VcXsrv is set to listen on display 0, run:

```bash
& "$env:ProgramFiles\VcXsrv\vcxsrv.exe" :0 -multiwindow -clipboard -wgl -ac # Runs # X server
docker desktop start; # Starts docker desktop
docker run --rm `
  -e DISPLAY=host.docker.internal:0 ` # Sets the GUI to run 
   # on the XLaunch client window 0
  -p 30001-30004:30001-30004 ` # Sets up ports
  -p 29999:29999 ` # Sets up ports
  universalrobots/ursim_e-series # Runs UR Sim and opens ports for UR RTDE
```

This command tells the container where to send its GUI.

## 6. Troubleshooting

- **No GUI Display?**  
  - Make sure VcXsrv is running and accepting connections. If not, open **XLaunch**.
  - Check that your firewall isn't blocking the connection.
  - Verify that the display number (`0`) matches your VcXsrv configuration.

- **Port Mapping Issues?**  
  The container might require additional port mappings depending on how URSim is set up. You can use `-p hostPort:containerPort` to forward ports if needed.

---

Follow these steps, and you should have URSim up and running on Windows 10!
