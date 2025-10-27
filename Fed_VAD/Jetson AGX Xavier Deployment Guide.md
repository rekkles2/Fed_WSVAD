
# Jetson AGX Xavier Deployment Guide for Federated Learning via Flower Framework
> [!NOTE]  
> Please **strictly follow the versions provided in this guide**, as the Jetson Xavier platform has a unique hardware architecture. It requires specific versions of PyTorch and related packages. Note that the communication package `grpcio` may cause version conflicts with other deep-learning packages if mismatched.

- [ ] **Jetson Xavier System Version:** `Ubuntu 20.04.6 LTS with JetPack 5.1-b147`
- [ ] **Flower Package Version:** `1.8.0`
- [ ] **PyTorch Version:** `1.12.0a0 + 02fb0b0f.nv22.06`
- [ ] **Docker Image Version:** `l4t-pytorch:r35.1.0-pth1.13-py3`
- [ ] **gRPC Version:** `grpcio==1.62.1`

<p align="center">
  <img src="https://github.com/rekkles2/Fed_WSVAD/raw/main/Figure/Jetson.jpg"
       alt="Figure 1. NVIDIA Jetson AGX Xavier."
       width="30%"/>
  <br>
  <em>Figure 1: NVIDIA Jetson AGX Xavier.</em>
</p>

---

## Let’s Start!!!

### 1. Prepare a Virtual Machine to Flash the AGX Xavier System

> [!NOTE]  
> The virtual machine should run **Ubuntu 18.04.6 LTS**, and the disk must have **at least 70 GB** of free space to download and flash the AGX Xavier system image.

There are many online tutorials for setting up virtual machines.  
I recommend using **VMware Workstation Pro 16** for its stability and ease of use.  
Since this guide focuses primarily on **flashing the AGX Xavier and preparing it for federated learning**, the details of virtual machine configuration are omitted here — these setup steps are widely documented and do not involve many hidden pitfalls.

---

### 2. Flash the System for AGX Xavier

<p align="center">
  <table>
    <tr>
      <td align="center" width="50%">
        <img src="https://github.com/rekkles2/Fed_WSVAD/raw/main/Figure/sdk.png"
             alt="Figure 2. NVIDIA SDK Manager version." width="95%"/>
        <br>
        <em>Figure 2. NVIDIA SDK Manager (version view).</em>
      </td>
      <td align="center" width="50%">
        <img src="https://github.com/rekkles2/Fed_WSVAD/raw/main/Figure/platform.png"
             alt="Figure 3. Platform selection in SDK Manager." width="95%"/>
        <br>
        <em>Figure 3. Platform selection in SDK Manager.</em>
      </td>
    </tr>
  </table>
</p>

Download **SDK Manager** [`sdkmanager_2.0.0-11402_amd64.deb`](https://github.com/rekkles2/Fed_WSVAD/releases/download/v1.0.0/sdkmanager_2.0.0-11402_amd64.deb) and transfer the package to your virtual machine.

Install the package:
```bash
sudo apt install ./sdkmanager_2.0.0-11402_amd64.deb
````

Launch SDK Manager (you should see a UI similar to **Figure 3**):

```bash
sdkmanager
```

When flashing begins, connect your host computer to the **AGX Xavier** via **USB**.
In SDK Manager, select **JetPack 5.1 (rev. 1)** for installation.

After the flashing process is complete, the **AGX Xavier** device will automatically power on and its display will light up — indicating that the system installation has finished successfully.
You can now proceed to the next step.

---

### 3. Expand the Storage Space on AGX Xavier

> [!NOTE]
> By default, the **AGX Xavier** provides only **32 GB** of onboard storage, which is insufficient for large datasets.
> After completing the initial flashing process, it is recommended to **expand the available storage** by **mounting and migrating the system to an external SSD**.

<p align="center">
  <table>
    <tr>
      <td align="center" width="33%">
        <img src="https://github.com/rekkles2/Fed_WSVAD/raw/main/Figure/disk.png"
             alt="Figure 4. Disk Utility app on Jetson AGX Xavier."
             width="95%"/>
        <br>
        <em>Figure 4. Disk Utility App.</em>
      </td>
      <td align="center" width="33%">
        <img src="https://github.com/rekkles2/Fed_WSVAD/raw/main/Figure/ssd.png"
             alt="Figure 5. Selecting the SSD for mounting."
             width="95%"/>
        <br>
        <em>Figure 5. Select SSD.</em>
      </td>
      <td align="center" width="33%">
        <img src="https://github.com/rekkles2/Fed_WSVAD/raw/main/Figure/ssd2.png"
             alt="Figure 6. Creating a new disk partition."
             width="95%"/>
        <br>
        <em>Figure 6. Create a Disk Partition.</em>
      </td>
    </tr>
  </table>
</p>

First, open the **Disks** application (as shown in **Figure 4**).
Locate the installed SSD and select it (**Figure 5**).
Press **Ctrl + D** to **format** the SSD and prepare it for mounting.

> 💡 *Note:* The demonstration device used in this guide was already flashed and configured,
> so the post-formatting screen is not displayed here.

Next, click the “+” button (**Figure 6**) to **create a new partition**, set a **volume name**, and then click **Create**.

To **migrate the root filesystem** from eMMC to the SSD, open a terminal **within the SSD directory** and run:

<p align="center">
  <img src="https://github.com/rekkles2/Fed_WSVAD/raw/main/Figure/nvme.png"
       alt="Figure 7. NVMe setup terminal on AGX Xavier."
       width="70%"/>
  <br>
  <em>Figure 7. Setting up NVMe boot for AGX Xavier.</em>
</p>

```bash
git clone https://github.com/jetsonhacks/rootOnNVMe.git
cd rootOnNVMe
./copy-rootfs-ssd.sh
./setup-service.sh
```

Once all steps are complete, **reboot the AGX Xavier** to finalize the setup:

```bash
sudo reboot
```

---

### 4. Let the Fan Spin Up ☺

<p align="center">
  <table>
    <tr>
      <td align="center" width="50%">
        <img src="https://github.com/rekkles2/Fed_WSVAD/raw/main/Figure/jtop1.jpg"
             alt="Figure 8. Jtop graphical interface on AGX Xavier."
             width="95%"/>
        <br>
        <em>Figure 8. Jtop graphical interface.</em>
      </td>
      <td align="center" width="50%">
        <img src="https://github.com/rekkles2/Fed_WSVAD/raw/main/Figure/jtop2.jpg"
             alt="Figure 9. Jtop system monitor."
             width="95%"/>
        <br>
        <em>Figure 9. Jtop graphical interface.</em>
      </td>
    </tr>
  </table>
</p>

To enable and monitor the fan on AGX Xavier, install **jetson-stats**:

```bash
sudo pip3 install --no-cache-dir -v -U jetson-stats
```

If an error occurs during installation or when starting `jtop`, check the service log:

```bash
journalctl -u jtop.service
```

---

### 5. Set the Docker Environment for AGX Xavier

First, pull the NVIDIA PyTorch image (this step might take some time depending on your network speed):

```bash
sudo docker pull nvcr.io/nvidia/l4t-pytorch:r35.1.0-pth1.13-py3
```

Then, run and open the container:

```bash
sudo docker run -it \
  --shm-size=25g \
  --runtime nvidia \
  --name l4t-pth113 \
  nvcr.io/nvidia/l4t-pytorch:r35.1.0-pth1.13-py3 \
  /bin/bash
```

<p align="center">
  <table>
    <tr>
      <td align="center" width="33%">
        <img src="https://github.com/rekkles2/Fed_WSVAD/raw/main/Figure/docker1.png"
             alt="Figure 10. Docker terminal on AGX Xavier."
             width="95%"/>
        <br>
        <em>Figure 10. Docker terminal interface.</em>
      </td>
      <td align="center" width="33%">
        <img src="https://github.com/rekkles2/Fed_WSVAD/raw/main/Figure/docker2.jpg"
             alt="Figure 11. File path navigation inside Docker."
             width="95%"/>
        <br>
        <em>Figure 11. File navigation inside Docker.</em>
      </td>
      <td align="center" width="33%">
        <img src="https://github.com/rekkles2/Fed_WSVAD/raw/main/Figure/docker3.jpg"
             alt="Figure 12. Dataset storage location."
             width="95%"/>
        <br>
        <em>Figure 12. Dataset storage location.</em>
      </td>
    </tr>
  </table>
</p>

After launching the container (**Figure 10**), navigate as shown in **Figure 11** to access your filesystem.
Upload datasets or project files into the folder indicated in **Figure 12**:

```
Other Locations → Computer → var → lib → docker → overlay2 → [container-ID] → merged → root
```

> [!TIP]
> The container’s internal files will **reset when it restarts**.
> To preserve your environment, commit the current container to a new image.
> For example, save it as **flower**:

```bash
sudo docker commit [original_container_ID] flower
```

Next time, you can open it with:

```bash
sudo docker run -it --shm-size=25g --runtime nvidia flower /bin/bash
```

---

### 6. Install Flower Framework for Federated Learning

Now install the Flower framework in the Docker environment:

```bash
pip3 install scikit-learn
pip3 install matplotlib
pip3 install numpy
```

Then:

```bash
pip3 install flwr==1.8.0
```

Then:

```bash
pip3 install --no-cache-dir --force-reinstall -Iv grpcio==1.62.1
```

Now, all conditions are prepared. Thanks for following! ❤

