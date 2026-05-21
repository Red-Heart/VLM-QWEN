# VLM-QWEN
Source code for VLM Qwen VL 7B (AGX ORIN , OAK-D PRO WIDE CAMERA)  
   
VLM-Only Robotic Perception and Reasoning Pipeline  
Project Overview   

This project implements a Vision-Language Model (VLM)-based robotic perception and reasoning pipeline using:  
  
**NVIDIA Jetson AGX Orin   
ROS2 Humble   
OAK-D Pro Wide stereo depth camera   
Qwen2.5-VL-7B-Instruct   
RViz2 visualization**   
    
The system performs:        
   
Open-vocabulary object detection       
Bounding box generation        
Semantic scene understanding        
Navigation-aware reasoning       
Stereo depth fusion             
XYZ coordinate estimation         

The pipeline is designed as a VLM-only semantic perception architecture for embodied AI and robotics research.   
  
VLM-Only Pipeline Concept
   
Traditional robotic perception pipelines usually use:  
   
Camera → YOLO/Object Detector → Depth → Navigation  
  
This project explores a VLM-only approach:  
   
Camera → VLM → Bounding Boxes + Reasoning → Depth Fusion → Navigation Understanding   
   
Instead of relying on a dedicated detector like YOLO, the Vision-Language Model itself:  
  
-detects objects  
-generates bounding boxes  
-reasons about the scene  
-understands navigation risks  
-suggests robot actions  
   
The stereo depth camera is then used to:   
   
calculate metric depth   
estimate XYZ coordinates   
improve spatial awareness   
   
This research investigates whether modern VLMs can perform perception and reasoning together in a unified architecture.  
  
System Architecture  
OAK-D Pro Wide RGB Stream  
            ↓  
ROS2 Image Topic  
            ↓  
Qwen2.5-VL-7B VLM Node  
            ↓  
Bounding Boxes + Semantic Reasoning  
            ↓  
Depth Fusion Node  
            ↓  
XYZ Coordinate Estimation  
            ↓  
RViz2 Visualization + Structured Output  
  
  
Hardware Used   
_NVIDIA Jetson AGX Orin  
32GB LPDDR5  
2048 CUDA cores   
64 Tensor cores   
12-core ARM Cortex-A78AE CPU   
CUDA 12.6   
JetPack 6   
Ubuntu 22.04_    
   
Used for:
     
VLM inference  
ROS2 processing  
depth processing   
robotic reasoning   
    
   
_OAK-D Pro Wide  
Stereo AI depth camera  
Wide FOV RGB camera  
Stereo depth estimation  
Intel Movidius Myriad X  
ROS2 compatible_  
  
Used for:  
  
RGB image streaming  
stereo depth maps  
depth fusion  
XYZ coordinate estimation  
  
  
**Software Stack  **
Component	Version  
_Ubuntu	22.04  
ROS2	Humble   
JetPack	6   
CUDA	12.6   
PyTorch	CUDA-enabled   
Transformers	HuggingFace   
OpenCV	Python OpenCV  
RViz2	ROS2 visualization_   
