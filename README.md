# Multi-Agentic Automatic Number-Plate Recognition (ANPR)  

## Project Overview  
This project implements an AI-driven Automatic Number-Plate Recognition (ANPR) system, leveraging deep learning and computer vision. The system integrates multiple AI models, including YOLO for detection, GANs and VAEs for image enhancement, CNNs for character recognition, and AI verification using Llama 3.2 Vision-Instruct.  

Originally developed for academic purposes, this project is now maintained as a side project for research, experimentation, and portfolio development.  

## Team Members  
- [Alberto Calabrese](https://github.com/Albi1999)
- [Marlon Helbing](https://github.com/maloooon)
- [Daniele Virzì](https://github.com/danielevirzi)

## Key Features  
- End-to-end ANPR pipeline for automatic detection, enhancement, and recognition of license plates  
- YOLO-based license plate detection for fast and efficient object localization  
- Image enhancement using GANs and Variational Autoencoders (VAEs)  
- CNN and LeNet-based character classification for improved recognition accuracy  
- AI verification and correction using Llama 3.2 Vision-Instruct  
- Modular and scalable design for research and development  

## Pipeline Overview  

### License Plate Detection  
- YOLO (You Only Look Once) is used for detecting license plates in images  
- The detected plates are cropped and preprocessed for further refinement  

### Image Enhancement and Processing  
- U-Net and GAN-based models enhance the quality of license plate images  
- Variational Autoencoders (VAEs) refine images and reduce noise  

### Character Recognition (OCR + CNN + LeNet)  
- A character segmentation module isolates individual license plate characters  
- A LeNet-based CNN classifies and recognizes alphanumeric symbols  
- OCR algorithms extract the final license plate text  

### AI Validation and Correction  
- Llama 3.2 Vision-Instruct verifies OCR results against the original image  
- The system re-prompts itself if prediction confidence is low, refining the output  

## Installation and Setup  

### Clone the Repository  
```bash
git clone https://github.com/Albi1999/multi-agent-anpr.git
cd multi-agent-anpr
```

### Install Dependencies  
```bash
pip install -r requirements.txt
```

### Run the Application  
```bash
python App.py
```

### Input an Image and Get the License Plate  
The system will detect, enhance, and recognize the license plate text automatically.  

## Future Improvements  
- Real-time video processing for live ANPR applications  
- Cloud deployment to make the system accessible as an online API  
- Multi-language OCR support to adapt to various license plate formats  
- Self-learning AI to improve OCR accuracy through reinforcement learning  

## License  
This project is open-source under the MIT License. See `LICENSE` for more information.

For contributions, suggestions, or inquiries, feel free to reach out.

</div>

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white&style=plastic" height="25"/>
  <img alt="Visual Studio Code" src="https://img.shields.io/badge/Visual Studio Code-007ACC?logo=VisualStudioCode&logoColor=white&style=plastic" height="25"/>
  <img alt="Google Colab" src="https://img.shields.io/badge/Google%20Colab-F9AB00?style=plastic&logo=googlecolab&logoColor=white&logoSize=auto" height="25"/>
  <img alt="Anaconda" src="https://img.shields.io/badge/Anaconda-44A833?style=plastic&logo=anaconda&logoColor=white&logoSize=auto" height="25"/>
  <img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-F37626?logo=Jupyter&logoColor=white&style=plastic" height="25"/>
</p>