
# Fed-WSVAD Guide 📚

This guide provides comprehensive instructions for setting up and running the **Federated Weakly Supervised Video Anomaly Detection (Fed-WSVAD)** framework. Follow these steps to successfully deploy and execute your federated learning experiments.

<p align="center">
  <img src="https://github.com/rekkles2/Fed_WSVAD/raw/main/Figure/Jetson.jpg" alt="Figure 1. NVIDIA Jetson AGX Xavier." width="40%"/>
  <br>
  <em>Figure 1: NVIDIA Jetson AGX Xavier.</em>
</p>



---

## 1. Preparation 🛠️

Before you begin, ensure all necessary hardware, software, and configurations are in place. Careful preparation is key to a smooth setup process.

### 1.1 Hardware Requirements 💻

* **Server:** You will need **one machine** (a standard PC or server) to host the central aggregation server.

* **Clients:** At least **two NVIDIA Jetson AGX Xavier** devices are required to function as the distributed clients.

* **Network:** A reliable **Wi-Fi network** is essential to connect the server and all client devices for communication. 🌐

### 1.2 Software & Data Setup 💾

* **Project Code & Dataset:** Ensure the required dataset and the project code directory (`Fed_WSVAD`) are successfully uploaded and available on **each** client (NVIDIA AGX Xavier) device.

### 1.3 Configuration Steps ⚙️

* **Server Address Modification:** Locate and modify the `server_address` variable within two specific files: `Fed_VAD/server.py` and `Fed_VAD/client_pytorch.py`.

* **Replace Placeholder IP:** Update the placeholder IP address with the actual **IPv4 address** of your server machine. The correct format should be `<YOUR_SERVER_IPv4_ADDRESS>:8080` (e.g., `192.168.1.100:8080`). This ensures clients can connect to the server. ✏️

---

## 2. Scene Similarity Analysis 📸

* **Purpose:** Prepare each scene photo (without humans), compute the similarity between scenes. The scene with the highest similarity might be chosen for paired experiments.

* **Execution:** Run the following script:

    ```bash
    python Fed_VAD/Scene_Similarity.py
    ```

---

## 3. Running the Framework ▶️

Once preparation is complete, follow these steps to initiate and run the federated learning framework.

### 3.1 Start the Server 🚀

Begin by starting the central server process on your designated server machine. Open a terminal and execute the following command:

```bash
# --rounds: This argument specifies the total number of federated learning rounds the training will perform.
python Fed_VAD/server.py --rounds=10
```

*This command starts the server, waiting for clients to connect and participate in **10 rounds** of federated learning.*

### 3.2 Start Each Client ✨

After the server is running, start the client process on **each** NVIDIA AGX Xavier device. It is crucial that each client is assigned a unique identification number (`--cid`). Open a terminal on each client device and run the following command:

```bash
# --cid: This argument assigns a unique Client ID to each participating device (e.g., 0, 1, 2, ...).
python Fed_VAD/client_pytorch.py --cid=<CLIENT_ID>
```

**Examples for Starting Clients:** 👇

* **Client 0:** Run `python Fed_VAD/client_pytorch.py --cid=0`

* **Client 1:** Run `python Fed_VAD/client_pytorch.py --cid=1`

* ...and continue this pattern for all additional clients you wish to include in the federated learning process.

---

By following these steps, you should be able to successfully set up and run the Fed-WSVAD framework. Good luck with your experiments! 👍
