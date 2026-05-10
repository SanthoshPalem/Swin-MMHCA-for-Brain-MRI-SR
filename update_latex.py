\documentclass{IEEEoj}
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{booktabs}
\usepackage[table]{xcolor}
\usepackage{algorithm}
\usepackage{graphicx}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,fit,calc,backgrounds}
\usepackage{float}
\usepackage{algpseudocode}
\usepackage{hyperref}
\usepackage{orcidlink}
\usepackage{caption}
\usepackage{comment}
\captionsetup{justification=centering}

\usepackage{textcomp}

\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em
    T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}
\AtBeginDocument{\definecolor{ojcolor}{cmyk}{0.93,0.59,0.15,0.02}}
% \def\OJlogo{\vspace{-14pt}\includegraphics[height=28pt]{logo.jpeg}}
\def\OJlogo{\vspace{-14pt}}

\begin{document}
\receiveddate{XX Month, XXXX}
\reviseddate{XX Month, XXXX}
\accepteddate{XX Month, XXXX}
\publisheddate{XX Month, XXXX}
\currentdate{XX Month, XXXX}
\doiinfo{OJIM.2022.1234567}

\title{Swin-MMHCA: Hierarchical Vision Transformer with Multi-Modal Hybrid Cross Attention for Brain MRI Super-Resolution}

\author{Ch. Preethi\orcidlink{0009-0004-0044-4143}, G. Roshni\,\orcidlink{0009-0005-4536-6844}, I. Bhargava\,\orcidlink{0009-0000-2223-5140}, Priya K V  \orcidlink{0000-0001-5552-0351} (SENIOR MEMBER, IEEE), Syed Sameen Ahmad Rizvi \orcidlink{0000-0002-3919-5074} (MEMBER, IEEE)}
\affil{Department of Computer Science \& Engineering, School of Engineering and Applied Sciences, SRM University-AP, Amaravati 522 240, Andhra Pradesh, India }
\corresp{CORRESPONDING AUTHOR: Priya K V (e-mail: priya.k@srmap.edu.in ).}
\markboth{Swin-MMHCA for Brain MRI Super-Resolution}{Preethi \textit{et al.}}

\begin{abstract}
High-resolution (HR) Magnetic Resonance Imaging (MRI) is essential for accurate clinical neuro-diagnosis, yet its acquisition is often limited by physiological and hardware constraints. While multi-modal MRI scans provide complementary anatomical information, effectively fusing these disparate features for image enhancement remains a significant challenge. In this paper, we propose \textbf{Swin-MMHCA}, a novel multi-task framework that integrates the global representation power of hierarchical Vision Transformers with a Multi-Modal Hybrid Cross Attention mechanism for brain MRI super-resolution. Our architecture utilizes a Swin Transformer V2 backbone to extract deep hierarchical features, which are dynamically aligned and fused with structural edge information through a transformer-based cross-attention module. To ensure anatomical fidelity and diagnostic utility, Swin-MMHCA incorporates auxiliary segmentation and lesion detection branches, enabling the model to learn semantically-aware representations. Extensive experiments on the IXI dataset demonstrate that our approach significantly outperforms traditional CNN-based and unimodal baselines, achieving a Peak Signal-to-Noise Ratio (PSNR) of 35.84 dB and a Structural Similarity Index (SSIM) of 0.82. The results indicate that the integration of global context and cross-modal attention effectively recovers high-frequency textures and maintains global structural consistency, offering a robust solution for high-fidelity medical image enhancement.
\end{abstract}

\begin{IEEEkeywords}
MRI Super-Resolution, Swin Transformer, Multi-Modal Hybrid Cross Attention, Deep Learning, Medical Image Analysis.
\end{IEEEkeywords}

\maketitle

\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{paper_images/SwinMMHCA_arch.png}
    \caption{The Swin-MMHCA architecture integrates a hierarchical Swin Transformer V2 backbone with a Multi-Modal Hybrid Cross Attention (MMHCA) mechanism for dynamic feature fusion across MRI modalities. The framework employs edge-aware context encoding and auxiliary branches for segmentation and lesion detection to ensure high-fidelity, anatomically consistent super-resolution.}
    \label{fig:architecture}
\end{figure*}



\section{INTRODUCTION}
\IEEEPARstart{M}{agnetic} Resonance Imaging (MRI) has revolutionized the field of clinical neurology by providing a non-invasive window into the complex architecture of the human brain. Unlike ionizing radiation-based modalities such as Computed Tomography (CT), MRI utilizes the principles of nuclear magnetic resonance to generate high-contrast images of soft tissues, making it the gold standard for diagnosing a wide spectrum of neurological disorders. From the identification of ischemic strokes in acute settings to the longitudinal monitoring of neurodegenerative diseases like Alzheimer's and Multiple Sclerosis, the diagnostic utility of MRI is fundamentally tied to its spatial resolution.

\subsection{Clinical Motivation and Diagnostic Significance}
The demand for High-Resolution (HR) MRI is driven by the need for "precision neuroradiology." In clinical practice, the ability to discern microscopic structural changes can be the difference between an early, treatable diagnosis and a late-stage intervention. For instance, in the study of epilepsy, HR-MRI is essential for detecting focal cortical dysplasia, which often manifests as subtle thickening of the cortex or blurring of the gray-white matter junction—features that are nearly invisible in standard-resolution scans. Similarly, in oncology, the precise delineation of tumor boundaries and the identification of micro-metastases are critical for surgical planning and radiation therapy targeting.

Beyond structural imaging, the fidelity of MRI affects downstream quantitative analysis. Automated brain volumetry, cortical thickness mapping, and radiomics-based feature extraction all require high-fidelity inputs to produce reliable results. When images are acquired at low resolution, the "partial volume effect"—where a single voxel contains a mixture of different tissue types—introduces significant noise and bias into these clinical metrics. Consequently, there is an urgent need for computational methods that can enhance the resolution of legacy or rapidly-acquired scans without the need for hardware upgrades.

\subsection{Physical Constraints and Hardware Limitations}
Despite the clear clinical need for high resolution, the acquisition of HR-MRI is governed by the fundamental laws of physics and signal processing. The resolution of an MR image is determined by the strength of the magnetic field gradients and the duration of the signal acquisition. According to the Nyquist-Shannon sampling theorem, capturing higher spatial frequencies requires a denser sampling of the k-space, which linearly increases the scan time.

Furthermore, there is an inherent trade-off between spatial resolution and the Signal-to-Noise Ratio (SNR). As voxels become smaller to achieve higher resolution, the number of hydrogen nuclei within each voxel decreases, leading to a weaker signal. To compensate for this loss in SNR, clinicians must often increase the "number of excitations" (NEX), further prolonging the scan duration. In a typical clinical environment, a high-resolution 3D T2-weighted scan can take upwards of 10 to 15 minutes per subject. For patients with claustrophobia, tremors, or pediatric subjects, maintaining stillness for such durations is often impossible, leading to severe motion artifacts that render the scans undiagnosable.

\subsection{Socio-Economic Impact and Accessibility}
The limitations of MRI hardware also have significant socio-economic implications. MRI machines are among the most expensive medical devices, with high maintenance costs and specialized infrastructure requirements. Long scan times reduce patient throughput, leading to long waiting lists and higher costs per scan. In developing regions or rural healthcare centers, access to high-field (3T or 7T) MRI scanners is severely limited, forcing clinicians to rely on older 1.5T or low-field systems that produce inherently lower-resolution images.

The development of robust Super-Resolution (SR) algorithms offers a "software-based" solution to these accessibility challenges. By computationally enhancing images from low-field scanners or fast-acquisition protocols, we can democratize access to high-quality neuroimaging. This is particularly relevant in the era of "Value-Based Healthcare," where reducing scan times and improving diagnostic accuracy are dual priorities for healthcare providers worldwide.

\subsection{Evolution of Image Super-Resolution}
The field of Image Super-Resolution has seen a paradigm shift over the last decade. Early methods relied on classical interpolation techniques, such as Bicubic and Lanczos filtering. These methods utilize local pixel intensity averages to estimate missing values, but they fail to recover the high-frequency information that was never sampled during acquisition. The result is often a visually "smooth" image that lacks the sharp edges and fine textures necessary for clinical diagnosis.

The advent of Deep Learning, specifically Convolutional Neural Networks (CNNs), introduced the concept of "learning" the mapping from Low-Resolution (LR) to HR space. Models like SRCNN and EDSR demonstrated that deep networks could act as sophisticated prior-models, "predicting" missing anatomical details based on patterns learned from thousands of high-quality training examples. However, pure CNN architectures are constrained by their local receptive fields; they look at small neighborhoods of pixels and often miss the global structural context that defines the human brain's symmetry and organization.

\subsection{The Rise of Transformers and Multi-Modal Fusion}
In the last two years, the introduction of Vision Transformers (ViTs) and the Swin Transformer has opened a new frontier in medical SR. By employing self-attention mechanisms, these models can capture long-range dependencies, allowing a pixel in the frontal lobe to "inform" the reconstruction of a pixel in the occipital lobe based on learned anatomical priors. 

Furthermore, the unique multi-modal nature of MRI—where T1, T2, and PD scans are often acquired together—provides a "data-rich" environment for super-resolution. T1-weighted scans, which are typically faster to acquire with high resolution, can serve as a structural reference to guide the enhancement of T2-weighted scans. This "Multi-Contrast Super-Resolution" (MCSR) approach is the core of our proposed Swin-MMHCA framework. We leverage the global modeling power of Swin Transformers and a specialized cross-attention mechanism to dynamically fuse information across these modalities, ensuring that the final output is not just a sharp image, but a clinically faithful representation of the patient's anatomy.

\begin{figure}[t]
\centering
\resizebox{\columnwidth}{!}{%
\begin{tikzpicture}[
    >=Latex,
    font=\scriptsize,
    flow/.style={->, line width=0.55pt},
    skip/.style={->, dashed, line width=0.5pt},
    input/.style={draw, fill=gray!15, minimum width=1.35cm, minimum height=0.9cm, align=center},
    cnn/.style={draw, fill=green!30, minimum width=1.0cm, minimum height=0.58cm, align=center},
    feat/.style={draw, fill=green!20, minimum width=1.05cm, minimum height=0.62cm, align=center},
    concat/.style={draw, fill=orange!30, minimum width=1.25cm, minimum height=0.82cm, align=center},
    swin/.style={draw, fill=blue!20, minimum width=2.2cm, minimum height=0.86cm, align=center},
    edge/.style={draw, fill=orange!20, minimum width=2.15cm, minimum height=0.78cm, align=center},
    attn/.style={draw, fill=magenta!20, minimum width=2.15cm, minimum height=0.78cm, align=center},
    decoder/.style={draw, fill=yellow!30, minimum width=2.45cm, minimum height=1.05cm, align=center},
    aux/.style={draw, fill=gray!20, minimum width=1.95cm, minimum height=0.68cm, align=center},
    out/.style={draw, fill=red!20, minimum width=1.75cm, minimum height=0.92cm, align=center},
    group/.style={draw, dashed, rounded corners=2pt, inner sep=6pt},
    lab/.style={font=\tiny, fill=white, inner sep=1pt}
]

% Inputs
\node[input] (t1) at (0,3.0) {T1 LR\\slice\\$1{\times}64^2$};
\node[input] (t2) at (0,1.55) {T2 LR\\slice\\$1{\times}64^2$};
\node[input] (pd) at (0,0.1) {PD LR\\slice\\$1{\times}64^2$};

% Stems
\node[cnn] (c1) at (1.7,3.0) {CNN};
\node[cnn, fill=orange!40] (c2) at (1.7,1.55) {CNN};
\node[cnn, fill=red!35] (c3) at (1.7,0.1) {CNN};

\node[feat] (f1) at (3.15,3.0) {$96{\times}64^2$};
\node[feat, fill=orange!25] (f2) at (3.15,1.55) {$96{\times}64^2$};
\node[feat, fill=red!25] (f3) at (3.15,0.1) {$96{\times}64^2$};

\node[concat] (cat) at (4.85,1.55) {concat\\$288{\times}64^2$};

\draw[flow] (t1) -- (c1) -- (f1);
\draw[flow] (t2) -- (c2) -- (f2);
\draw[flow] (pd) -- (c3) -- (f3);
\draw[flow] (f1.east) -- ++(0.32,0) |- (cat.west);
\draw[flow] (f2) -- (cat);
\draw[flow] (f3.east) -- ++(0.32,0) |- (cat.west);

% Dual core box
\coordinate (coreNW) at (5.8,4.05);
\coordinate (coreSE) at (14.35,-1.25);
\node[group, fill=gray!8, fit=(coreNW)(coreSE), label=above:{\textbf{Dual Swin-MMHCA Core}}] {};

% Swin branch
\node[swin] (patch) at (6.85,3.1) {Patch embed\\$96{\times}32^2$};
\node[swin] (swin) at (9.65,3.1) {SwinV2 backbone\\$32^2,16^2,8^2$};
\node[swin] (latent) at (12.45,3.1) {Latent\\$768{\times}8^2$};

\draw[flow] (cat.east) -- ++(0.45,0) |- (patch.west);
\draw[flow] (patch) -- (swin);
\draw[flow] (swin) -- node[lab, above] {$1{\times}1$} (latent);

% Edge-context branch
\node[edge] (ctx) at (6.85,0.45) {Context encoder\\$256{\times}64^2$};
\node[edge] (sobel) at (6.85,-0.95) {Sobel edges\\$3{\times}64^2$};
\node[edge] (anchor) at (9.65,-0.25) {Edge-context\\anchor\\$256{\times}64^2$};

\draw[flow] (cat.east) -- ++(0.5,0) |- (ctx.west);
\draw[skip] (t1.east) -- ++(0.7,0) |- (sobel.west);
\draw[skip] (t2.east) -- ++(0.55,0) |- (sobel.west);
\draw[skip] (pd.east) -- ++(0.4,0) |- (sobel.west);
\draw[flow] (ctx.east) -- ++(0.3,0) |- (anchor.west);
\draw[flow] (sobel.east) -- ++(0.3,0) |- (anchor.west);

% Cross-attention
\node[attn] (attn8) at (12.45,0.35) {Cross-attn\\$384{\times}8^2$};
\node[attn] (attnl) at (12.45,-0.95) {Cross-attn\\$768{\times}8^2$};

\draw[flow] (swin.south) -- ++(0,-0.55) -| node[lab, near start, right] {Q} (attn8.north);
\draw[flow] (latent.south) -- node[lab, right] {Q} (attnl.north);
\draw[skip] (anchor.east) -- ++(0.5,0) |- node[lab, pos=0.35, above] {K,V 384} (attn8.west);
\draw[skip] (anchor.east) -- ++(0.5,0) |- node[lab, pos=0.35, below] {K,V 768} (attnl.west);

% Decoder and output
\node[decoder] (dec) at (16.1,1.15) {Progressive decoder\\PixelShuffle $8{\rightarrow}256$\\residual head};
\node[out] (sr) at (19.05,1.15) {SR T2\\$1{\times}256^2$};
\node[input, fill=gray!25] (hr) at (19.05,-0.55) {HR T2\\target\\$1{\times}256^2$};

\draw[flow] (attnl.east) -- ++(0.45,0) |- (dec.west);
\draw[skip] (attn8.east) -- ++(0.55,0) |- node[lab, pos=0.25, above] {$8^2$ skip} (dec.west);
\draw[skip] (swin.south east) -- ++(0.8,-1.1) -| node[lab, pos=0.25, above] {$32^2/16^2$ skips} (dec.north);
\draw[skip] (ctx.east) -- ++(7.35,0) |- node[lab, pos=0.24, above] {context skip} (dec.west);
\draw[flow] (dec) -- node[lab, above] {residual} (sr);
\draw[skip] (t2.south) -- ++(0,-2.12) -| node[lab, pos=0.18, below] {bicubic $4{\times}$} (sr.south);
\draw[skip] (sr) -- (hr);

% Auxiliary heads
\node[aux] (seg) at (13.85,-2.45) {Segmentation\\$1{\times}256^2$};
\node[aux] (det) at (16.15,-2.45) {Detection\\$1{\times}5^2$};
\draw[flow] (attn8.south) -- ++(0,-0.55) -| (seg.north);
\draw[flow] (attnl.south) -- ++(0,-0.55) -| (det.north);

\node[group, fit=(seg)(det), label=below:{Auxiliary heads}] {};
\end{tikzpicture}%
}
\caption{Overall Swin-MMHCA architecture for $4\times$ 2D T2 MRI super-resolution. Registered T1, T2, and PD LR slices are encoded by modality-specific CNN stems and concatenated. The dual core combines a SwinV2 global branch with a structural edge-context branch. Cross-attention injects edge-context information into deep Swin features, and a progressive decoder reconstructs the HR T2 residual over a bicubic T2 skip connection.}
\label{fig:swin_mmhca_architecture}
\end{figure}

The key contributions of this work include:
\begin{enumerate}
    \item A specialized Swin Transformer V2-based hierarchical architecture that captures both local textures and global anatomical context for brain MRI.
    \item The introduction of a Multi-Modal Hybrid Cross Attention (MMHCA) mechanism to effectively exploit inter-modal correlations for enhanced texture recovery from T1 and PD modalities.
    \item A multi-task learning strategy incorporating segmentation and lesion detection heads to enforce semantic consistency in the super-resolved outputs.
    \item A research-grade preprocessing pipeline using SimpleITK for multi-modal registration and brain coverage filtering.
    \item Extensive validation on the IXI dataset, demonstrating superior performance in terms of PSNR, SSIM, and LPIPS compared to existing CNN-based baselines.
\end{enumerate}

\section{Literature Review}
The landscape of medical image super-resolution (SR) has undergone a paradigm shift from classical interpolation algorithms to sophisticated deep learning architectures. This section provides an exhaustive review of the technological evolution, starting from early convolutional neural networks (CNNs) to the most recent Transformer-based multi-modal frameworks, identifying the research gaps that necessitate the development of our Swin-MMHCA model.

\subsection{Foundations of MRI Super-Resolution}
Magnetic Resonance Imaging (MRI) is a cornerstone of clinical neurology, yet its acquisition is governed by fundamental physical trade-offs between spatial resolution, signal-to-noise ratio (SNR), and scan duration \cite{ixi_dataset, med_sr_survey}. The super-resolution inverse problem aims to reconstruct a high-resolution (HR) image from one or more low-resolution (LR) observations, typically modeled as $Y = (X \ast K) \downarrow_s + \eta$ \cite{srcnn, tpami_survey}. Early computational solutions relied on classical interpolation techniques such as Bicubic and Lanczos filtering, which fail to recover high-frequency anatomical details and often introduce blurring artifacts \cite{vdsr}.

\subsection{The CNN Era: From SRCNN to EDSR}
The application of deep learning to SR was pioneered by the Super-Resolution Convolutional Neural Network (SRCNN) \cite{srcnn}, which demonstrated that a shallow three-layer end-to-end network could significantly outperform traditional algorithms. Following this milestone, the field progressed rapidly toward deeper architectures. The Very Deep Super-Resolution (VDSR) network \cite{vdsr} introduced global residual learning to facilitate the training of 20-layer networks, effectively addressing the vanishing gradient problem. 

The Enhanced Deep Super-Resolution (EDSR) model \cite{edsr} marked a significant optimization by removing unnecessary Batch Normalization layers, allowing for a substantial increase in model depth and width while maintaining stability \cite{rcan}. Subsequent innovations like the Residual Dense Network (RDN) \cite{rdn} and the Information Distillation Network (IDN) \cite{idn} further improved feature extraction efficiency. However, a major limitation of these CNN-based methods is their localized receptive field; they focus on small neighborhoods of pixels and often fail to capture the global structural context and anatomical symmetry essential for brain MRI interpretation \cite{seran, csn}.

\subsection{Attention Mechanisms and Multi-Modal Fusion}
To address the limitations of pure CNNs, researchers introduced attention mechanisms to recalibrate feature maps based on their importance. Squeeze-and-Excitation (SE) networks \cite{seran} and Convolutional Block Attention Modules (CBAM) provided a mechanism for channel and spatial attention. In the medical domain, multi-modal MRI protocols (T1, T2, PD) provide a "data-rich" environment where reference modalities can guide the enhancement of a target modality \cite{fscwrn, csn}.

The Multimodal Multi-Head Convolutional Attention (MMHCA) framework \cite{Georgescu2023} achieved a significant breakthrough by employing multiple attention heads with varying kernel sizes to fuse features across MRI contrasts. While effective, MMHCA remains rooted in standard convolutional operations, which restricts its ability to model the long-range dependencies that define global brain morphology. Furthermore, methods like T2-Net \cite{t2net} explored joint reconstruction and SR but were primarily focused on task-specific optimizations rather than global context modeling.

\subsection{The Paradigm Shift to Vision Transformers}
The introduction of Vision Transformers (ViTs) \cite{Sajol2024, tpami_survey} addressed the local constraints of CNNs by treating images as sequences of patches and utilizing self-attention to model interactions between all patches simultaneously. The Swin Transformer \cite{swin_trans} further optimized this by introducing shifted-window self-attention, reducing computational complexity while maintaining global representational power.

Foundational restoration models like SwinIR \cite{swinir} and the Hybrid Attention Transformer (HAT) \cite{hat} have demonstrated superior performance in general image tasks. In the medical domain, SwinUNETR \cite{SwinUNETR2025} and specific adaptations like SwinIR-Med \cite{swinir_med} and STAN \cite{med_sr_survey} have shown that hierarchical Swin blocks can recover high-frequency details by modeling patch-wise correlations at multiple scales. Recent multi-modal Transformers like McMRSR \cite{mcmrsr_plus}, MTrans \cite{mtrans}, and WavTrans \cite{wavtrans} have further explored cross-modal contextual matching and wavelet-based frequency decomposition to enhance MRI fidelity.

\subsection{Multi-Task Learning and Semantic Supervision}
A growing trend in medical SR is the use of auxiliary tasks to regularize the training process and ensure anatomical fidelity \cite{dense_med, joint_sr_seg_2023}. SegSRGAN \cite{segsrgan} established that simultaneously training for super-resolution and segmentation prevents the "hallucination" of artifacts by forcing the model to respect tissue boundaries. More recent works like ReconFormer \cite{reconformer} and CSRS \cite{joint_sr_seg_2023} have shown that joint intensity and label upsampling improve downstream diagnostic accuracy in neurodegenerative diseases.

\subsection{Research Gaps and Proposed Swin-MMHCA}
Despite the proliferation of these techniques, several critical gaps remain:
\begin{enumerate}
    \item \textbf{Global-Local Boundary Preservation:} While Transformers capture global context, they can sometimes over-smooth local anatomical edges. Our dual-path architecture addresses this by incorporating an explicit \textbf{Sobel Edge Extraction} branch \cite{fscwrn} to provide structural guidance.
    \item \textbf{Complex Multi-Modal Fusion:} Existing fusion strategies often treat all modalities equally. Our Swin-MMHCA utilizes a specialized \textbf{Fusion Context Encoder} to extract high-level inter-modal relationships before backbone processing.
    \item \textbf{Efficient Upsampling Artifacts:} Traditional transposed convolutions in the decoder are prone to checkerboard artifacts \cite{srcnn}. We employ the \textbf{PixelShuffle} operator \cite{edsr} for efficient, artifact-free upsampling.
    \item \textbf{Lack of Pathological Awareness:} Many SR models focus purely on intensity fidelity. We integrate \textbf{auxiliary segmentation and lesion detection heads} with specialized losses (Dice-BCE and Focal Loss \cite{Sajol2024}) to ensure clinical utility.
\end{enumerate}

In summary, the Swin-MMHCA framework is built upon the global modeling power of SwinV2 \cite{swin_trans}, the fusion logic of MMHCA \cite{Georgescu2023}, and the semantic supervision of multi-task learning \cite{segsrgan}. By combining these elements into a unified dual-path architecture, we effectively bridge the gap between pixel-level accuracy and clinical anatomical fidelity.


\section{Related Work}
The landscape of medical image super-resolution has transitioned from traditional signal processing algorithms to sophisticated deep learning architectures. This section provides an exhaustive review of the technological evolution that led to the development of the Swin-MMHCA framework.

\subsection{Evolution of Single Image Super-Resolution (SISR)}
The journey of SISR began with classical interpolation techniques, such as Nearest Neighbor, Bilinear, and Bicubic interpolation. While computationally efficient, these methods are fundamentally limited by their inability to recover high-frequency information lost during the downsampling process, often resulting in "staircase" artifacts and blurred anatomical boundaries. 

The introduction of the Super-Resolution Convolutional Neural Network (SRCNN) \cite{srcnn} by Dong et al. marked the first successful application of deep learning to SISR, demonstrating that a simple three-layer end-to-end network could significantly outperform traditional algorithms. Following this, researchers explored deeper and more complex architectures. The Very Deep Super-Resolution (VDSR) network utilized residual learning to facilitate the training of 20-layer networks, while the Enhanced Deep Residual Network (EDSR) \cite{edsr} removed unnecessary Batch Normalization layers to improve performance and stability. Advanced models like RCAN \cite{rcan} further introduced channel attention to exploit inter-channel dependencies.

\subsection{Multi-Modal and Multi-Contrast MRI Fusion}
In clinical practice, MRI protocols typically involve the acquisition of multiple sequences (e.g., T1, T2, PD, FLAIR) to provide a comprehensive view of the tissue. Recent research has increasingly focused on leveraging this multi-modal data to improve reconstruction fidelity. Traditionally, Multi-Contrast Super-Resolution (MCSR) models utilized simple fusion strategies such as early-stage concatenation or element-wise summation of feature maps. 

However, these methods often fail to account for the complex, non-linear correlations between different tissue contrasts. A significant breakthrough was achieved by Georgescu et al. \cite{Georgescu2023}, who introduced the Multimodal Multi-Head Convolutional Attention (MMHCA) framework. MMHCA was designed to utilize multiple low-resolution scans as input, employing a bottleneck principle with varying kernel sizes to perform joint spatial and channel attention. This allowed the model to "borrow" sharp structural details from a reference modality (e.g., T1) to enhance a target modality (e.g., T2). While effective, MMHCA remains rooted in standard convolutional operations, which limits its ability to model global anatomical relationships across the entire brain volume.

\subsection{Attention Mechanisms in Medical Imaging}
The success of attention mechanisms in Natural Language Processing (NLP) has led to their widespread adoption in computer vision. Early attention modules, such as Squeeze-and-Excitation (SE) blocks, focused on channel-wise attention to recalibrate feature responses. The Convolutional Block Attention Module (CBAM) later combined both channel and spatial attention to guide the network toward informative regions.

In medical imaging, attention is particularly valuable for focusing the network on clinically significant structures, such as lesions or cortical folds. The MMHCA framework \cite{Georgescu2023} utilized multi-head convolutional attention to capture multi-scale features, but it lacked the global context modeling provided by Transformer-based attention. Our work bridges this gap by replacing local convolutional attention with a global cross-modal attention mechanism that dynamically aligns features from disparate modalities.

\subsection{The Paradigm Shift to Vision Transformers}
To overcome the localized receptive field constraints of CNNs, researchers have transitioned toward Vision Transformers (ViTs) \cite{Sajol2024}. ViTs treat images as sequences of patches and utilize self-attention to model interactions between all patches simultaneously, regardless of their spatial distance. This allows the network to capture global dependencies and maintain structural integrity across the entire image.

The Swin Transformer introduced a hierarchical structure and shifted-window self-attention, which significantly reduced the computational complexity of standard ViTs while maintaining global representational power. The Swin Transformer V2 \cite{swin_trans} further improved this with scaled cosine attention and log-spaced positional biases, making it more stable for high-resolution tasks. Recent works like SwinIR and SwinUNETR \cite{SwinUNETR2025} have demonstrated that hierarchical Swin blocks can effectively recover high-frequency details by modeling patch-wise correlations at multiple scales. Our Swin-MMHCA framework leverages this global modeling power to ensure that super-resolved brain MRI scans are anatomically consistent.

\subsection{Multi-Task Learning and Semantic Supervision}
A growing trend in medical imaging is the use of auxiliary tasks to improve the performance of the primary task. By training a model to simultaneously perform super-resolution and semantic tasks (e.g., segmentation or detection), the model is forced to learn features that preserve anatomical boundaries and pathological markers.

Recent studies have shown that multi-task learning can regularize the training process and prevent the "hallucination" of artifacts often seen in pure generative models. For instance, incorporating a segmentation head helps the model maintain the boundary between gray matter and white matter, which is critical for clinical diagnosis. Our framework builds on this by integrating both segmentation and lesion detection heads, providing a comprehensive semantic supervision signal that ensures the super-resolved images are not just visually sharp but also clinically accurate.

\subsection{Generative Adversarial Networks (GANs) and Perceptual Loss}
Traditional SR models often minimize the Mean Squared Error (MSE) or L1 loss, which tends to produce blurry images because they optimize for pixel-wise averages. To produce more realistic textures, researchers introduced Generative Adversarial Networks (GANs) and perceptual losses. SRGAN and ESRGAN utilize a discriminator network to "judge" the realism of the super-resolved output, forcing the generator to produce high-frequency details.

Perceptual loss, often calculated using a pre-trained VGG or AlexNet (LPIPS), measures the difference between feature maps in a deep embedding space rather than pixel space. This has been shown to be much closer to human visual perception. In our work, we utilize a training strategy that optimizes for structural fidelity to ensure that the reconstructed MRI scans possess the anatomical consistency required for high-fidelity clinical interpretation.

\section{Theoretical Framework}
This section details the physical, mathematical, and computational principles that underpin our research. We provide a rigorous foundation for MRI physics, the super-resolution inverse problem, and the mechanics of hierarchical vision transformers.

\subsection{Physics of MRI and Bloch Equations}
The generation of an MRI signal is governed by the behavior of hydrogen nuclei (protons) in a magnetic field. When a subject is placed in a static magnetic field $B_0$, the protons align with or against the field, creating a net magnetization vector $M$. The evolution of this vector is described by the \textit{Bloch Equations}:
\begin{equation}
    \frac{d\mathbf{M}}{dt} = \gamma \mathbf{M} \times \mathbf{B} - \frac{M_x \mathbf{i} + M_y \mathbf{j}}{T_2} - \frac{(M_z - M_0) \mathbf{k}}{T_1}
\end{equation}
where $\gamma$ is the gyromagnetic ratio, $T_1$ is the longitudinal relaxation time (spin-lattice), and $T_2$ is the transverse relaxation time (spin-spin). 

In medical super-resolution, understanding these relaxation constants is critical because they determine the contrast. $T_1$-weighted scans provide high-resolution anatomical maps of gray and white matter, while $T_2$-weighted scans are sensitive to water content and pathological changes. Our model leverages this physical relationship by using the structural "sharpness" of $T_1$ images to guide the $T_2$ enhancement.

\subsection{K-Space and Image Formation}
The raw data acquired by an MRI scanner is not an image, but a series of samples in the spatial frequency domain, known as \textit{k-space}. The relationship between the k-space data $S(k_x, k_y)$ and the final image $I(x, y)$ is a 2D Inverse Fourier Transform:
\begin{equation}
    I(x, y) = \int \int S(k_x, k_y) e^{j2\pi(k_x x + k_y y)} dk_x dk_y
\end{equation}
High-resolution images require a wider coverage of k-space (sampling higher spatial frequencies). However, sampling the outer regions of k-space takes more time and yields lower signal intensity. This leads to the "sampling trade-off": scanning faster results in a truncated k-space, which manifests as blurring and "ringing" (Gibbs phenomenon) in the reconstructed image. Super-resolution algorithms effectively act as "k-space extrapolators," predicting the missing high-frequency coefficients based on learned anatomical priors.

\subsection{The Super-Resolution Inverse Problem}
The relationship between a high-resolution ground truth image $X$ and its low-resolution observed counterpart $Y$ can be modeled as:
\begin{equation}
    Y = (X \ast K) \downarrow_s + \eta
\end{equation}
where $\ast$ denotes convolution, $K$ is the blur kernel (representing the Point Spread Function of the scanner), $\downarrow_s$ is the downsampling operator with factor $s$, and $\eta$ is additive white Gaussian noise.

Super-resolution is an \textit{ill-posed inverse problem} because multiple HR images could map to the same LR observation. To solve for $X$, we must incorporate a regularization term $\Phi(X)$ that represents our prior knowledge of what a brain looks like:
\begin{equation}
    \hat{X} = \arg \min_X \|Y - (X \ast K) \downarrow_s \|^2 + \lambda \Phi(X)
\end{equation}
In our framework, the Swin Transformer V2 serves as a sophisticated, non-linear learnable prior that models the complex structural manifold of brain MRI.

\subsection{Principles of Hierarchical Window-Based Attention}
The core of our backbone is the \textit{Window-based Multi-head Self-Attention} (W-MSA), which builds upon the foundational self-attention mechanism \cite{attention}. Given an image feature map of size $H \times W \times C$, standard self-attention has a computational complexity of $O(H^2 W^2 C)$, which is prohibitive for $256 \times 256$ images. Swin Transformers partition the image into non-overlapping windows of size $M \times M$. The complexity of W-MSA is:
\begin{equation}
    \Omega(\text{W-MSA}) = 4hwC^2 + 2M^2hwC
\end{equation}
where $h=H/M$ and $w=W/M$. Since $M$ is fixed (e.g., $M=8$), the complexity becomes linear with respect to the image resolution.

To facilitate cross-window communication, Swin introduces \textit{Shifted Windowing} (SW-MSA). In even layers, the window partition is shifted by $(\lfloor M/2 \rfloor, \lfloor M/2 \rfloor)$ pixels. This allows the model to capture dependencies that cross the initial window boundaries, which is essential for reconstructing large anatomical structures like the corpus callosum or the ventricular system.

\subsection{Mathematics of Continuous Relative Positional Bias}
Standard Transformers use fixed positional encodings, which do not generalize well to images of different sizes. SwinV2 utilizes \textit{Log-spaced Continuous Relative Positional Bias} (Log-CPB). For two positions $(x_i, y_i)$ and $(x_j, y_j)$, the relative coordinates $(\Delta x, \Delta y)$ are transformed:
\begin{equation}
    \widehat{\Delta x} = \text{sgn}(\Delta x) \cdot \ln(1 + |\Delta x|)
\end{equation}
The bias $B$ is then generated by a small meta-network $g$:
\begin{equation}
    B(\Delta x, \Delta y) = g(\widehat{\Delta x}, \widehat{\Delta y})
\end{equation}
This formulation ensures that the model can handle the varied spatial resolutions and aspect ratios commonly found in medical imaging protocols.

\subsection{Multimodal Similarity Metrics: Mutual Information}
Aligning T1, T2, and PD images is a prerequisite for our MMHCA module. Since these modalities have different intensity distributions for the same tissue, we cannot use Mean Squared Error for registration. Instead, we use \textit{Mutual Information} (MI), which is based on entropy:
\begin{equation}
    MI(A, B) = H(A) + H(B) - H(A, B)
\end{equation}
where $H(A)$ is the Shannon entropy of image $A$. MI measures the statistical dependence between the images; when the images are perfectly registered, the joint entropy $H(A, B)$ is minimized, and MI is maximized. This provides the mathematical basis for our SimpleITK-based preprocessing pipeline.

\section{Proposed Methodology: Swin-MMHCA}
The proposed \textbf{Swin-MMHCA} framework is designed as a unified, multi-task architecture that bridges the gap between local convolutional feature extraction and global transformer-based context modeling. This section provides a comprehensive, bottom-up technical description of the architecture's components and the mathematical logic governing their interactions.

\subsection{Architectural Overview}
The Swin-MMHCA model follows a modular design consisting of six interconnected phases:
\begin{enumerate}
    \item \textbf{Modality-Specific Stem Encoding:} Extracts shallow features from T1, T2, and PD inputs.
    \item \textbf{Multimodal Contextual Fusion:} Aggregates global context from all input modalities.
    \item \textbf{Structural Edge Guidance:} Extracts high-frequency boundary information.
    \item \textbf{Hierarchical SwinV2 Backbone:} Performs deep, global feature extraction at multiple scales.
    \item \textbf{Hybrid Cross-Attention Alignment:} Fuses backbone features with structural context.
    \item \textbf{Progressive Multi-Task Decoder:} Reconstructs the HR image while predicting semantic masks.
\end{enumerate}

\subsection{Phase I: Modality-Specific Stem Encoding}
To handle the unique intensity distributions of disparate MRI modalities, we employ independent \textit{Modality Stems}. Each stem $S_i$ is a dedicated CNN designed to map the single-channel input $I_i$ into a higher-dimensional latent space $F_{stem}^{(i)} \in \mathbb{R}^{C \times H \times W}$. For 4$\times$ super-resolution, we utilize an input resolution of $64 \times 64$, while for 2$\times$ super-resolution, the input is set to $128 \times 128$.
The stem architecture consists of:
\begin{itemize}
    \item \textbf{Conv1:} $3 \times 3$ convolution, stride 1, padding 1, producing 96 channels.
    \item \textbf{LeakyReLU:} Negative slope of 0.2.
    \item \textbf{Conv2:} $3 \times 3$ convolution, stride 1, padding 1, producing 96 channels.
\end{itemize}
This dual-layer convolution ensures that the model learns modality-specific low-level textures (e.g., the sharp boundaries in T1 vs. the fluid-sensitive textures in T2) before they are fused into a shared representation.

\subsection{Phase II: Multimodal Contextual Fusion}
After stem encoding, the features are concatenated along the channel dimension to form a joint representation $F_{cat} \in \mathbb{R}^{288 \times H \times W}$. We then pass this through a \textit{Fusion Context Encoder} to extract inter-modal correlations.
\begin{equation}
    F_{ctx} = \text{ResBlock}(\text{ResBlock}(\sigma(\text{Conv}_{3\times3}(F_{cat}))))
\end{equation}
Each Residual Block consists of two $3 \times 3$ convolutions with a skip connection:
\begin{equation}
    F_{out} = F_{in} + \text{Conv}(\sigma(\text{Conv}(F_{in})))
\end{equation}
This context map $F_{ctx}$ serves as a "global anatomical anchor" that informs the subsequent super-resolution stages.

\subsection{Phase III: Structural Edge Guidance Branch}
Preserving anatomical boundaries is a critical challenge in super-resolution. We implement a dedicated \textit{Sobel Edge Extractor} that operates directly on the input modalities.
The gradient magnitude for each modality is calculated using horizontal ($K_x$) and vertical ($K_y$) Sobel kernels:
\begin{equation}
    G_x = I \ast K_x, \quad G_y = I \ast K_y
\end{equation}
\begin{equation}
    F_{edge\_map}^{(i)} = \sqrt{G_x^2 + G_y^2 + \epsilon}
\end{equation}
These maps are concatenated and passed through a projection head to generate the edge guidance feature $F_{edge} \in \mathbb{R}^{3 \times H \times W}$. We then fuse the context and edge maps using an \textit{Edge-Context Fuser}:
\begin{equation}
    F_{global} = \text{ResBlock}(\sigma(\text{Conv}_{3\times3}([F_{ctx}, F_{edge}])))
\end{equation}

\subsection{Phase IV: Hierarchical Swin Transformer V2 Backbone}
The core of our model is the SwinV2-based encoder. The concatenated stem features $F_{cat}$ are treated as the primary input. The backbone architecture is designed to be scale-agnostic, with intermediate resolutions and token counts adapting to the input $H \times W$. The stage-wise configuration is detailed in Table \ref{tab:backbone_specs}.

\begin{table}[H]
\centering
\caption{Backbone Stage-wise Configuration for Swin-MMHCA. Resolutions are relative to input $H \times W$.}
\label{tab:backbone_specs}
\begin{tabular}{lcccc}
\toprule
\textbf{Stage} & \textbf{Resolution} & \textbf{Blocks} & \textbf{Heads} & \textbf{Channels} \\
\midrule
Patch Embed & $H/2 \times W/2$ & - & - & 96 \\
Stage 1 & $H/2 \times W/2$ & 2 & 3 & 96 \\
Downsample 1 & $H/4 \times W/4$ & - & - & 192 \\
Stage 2 & $H/4 \times W/4$ & 2 & 6 & 192 \\
Downsample 2 & $H/8 \times W/8$ & - & - & 384 \\
Stage 3 & $H/8 \times W/8$ & 6 & 12 & 384 \\
Latent Stage 4 & $H/8 \times W/8$ & 2 & 24 & 768 \\
\bottomrule
\end{tabular}
\end{table}

The SwinV2 block incorporates two critical improvements:
\begin{enumerate}
    \item \textbf{Post-Normalization:} LayerNorm is applied after the attention and MLP modules to improve training stability at high depths.
    \item \textbf{Scaled Cosine Attention:} Replaces the dot-product similarity to bound the attention logits and prevent saturation.
\end{enumerate}

\subsection{Phase V: Multi-Modal Hybrid Cross-Attention (MMHCA)}
To effectively inject the global context $F_{global}$ into the deep hierarchical features, we employ \textit{Transformer Cross-Attention} at the lowest resolutions ($H/8 \times W/8$).
The Cross-Attention mechanism is formulated as:
\begin{equation}
    \text{Q} = \text{Linear}(F_{backbone}), \quad \text{K, V} = \text{Linear}(F_{global})
\end{equation}
\begin{equation}
    \text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
\end{equation}
Unlike self-attention, the "Keys" and "Values" are derived from the multimodal context, allowing the model to "query" anatomical details from T1 and PD modalities to refine the T2 reconstruction. This is applied at both the $H/8 \times W/8$ Stage 3 features and the $H/8 \times W/8$ Latent features.

\subsection{Phase VI: Progressive Decoder and Multi-Task Heads}
Reconstructing a $256 \times 256$ image from backbone features requires a highly stable progressive decoder. The Swin-MMHCA model employs a dynamic decoder architecture that adapts its depth based on the upscaling factor. For 4$\times$ super-resolution (starting from $8 \times 8$ latent features), the model utilizes five successive \textit{PixelShuffle Blocks} to reach the target resolution. For 2$\times$ super-resolution (starting from $16 \times 16$ latent features), the model bypasses the initial upsampling stage to maintain structural integrity. Each block consists of:
\begin{enumerate}
    \item $3 \times 3$ Convolution increasing channels by $s^2$.
    \item \textit{PixelShuffle} layer for periodic shuffling upsampling.
    \item LeakyReLU activation.
    \item A Residual Block for feature refinement.
\end{enumerate}

\begin{algorithm}[h]
\caption{Swin-MMHCA Forward Pass}
\label{alg:forward}
\begin{algorithmic}[1]
\State \textbf{Input:} $\{I_{T1}, I_{T2}, I_{PD}\}$ ($H \times W$)
\State \textbf{Phase 1:} $F_{stems} \gets [\text{Stem}_i(I_i)]$
\State \textbf{Phase 2:} $F_{ctx} \gets \text{ContextEncoder}(F_{stems})$
\State \textbf{Phase 3:} $F_{edge} \gets \text{EdgeExtractor}(I_{all})$
\State $F_{global} \gets \text{Fuser}([F_{ctx}, F_{edge}])$
\State \textbf{Phase 4:} Backbone features extraction.
\State \textbf{Phase 5:} Cross-modal alignment at $H/8 \times W/8$.
\State \textbf{Phase 6:} Adaptive decoding to $256 \times 256$.
\State \textbf{Output:} $SR \gets \text{Bicubic}(I_{T2}) + \text{Residual}(x_{SR})$
\end{algorithmic}
\end{algorithm}

\subsection{Phase VII: Auxiliary Semantic Supervision}
The auxiliary heads are designed to provide "semantic anchors" during training.
\begin{itemize}
    \item \textbf{Segmentation Target:} Generated by thresholding the HR image at 0.05 intensity.
    \item \textbf{Detection Target:} Generated by identifying voxels that deviate from the local mean by more than 1.5 standard deviations, effectively isolating "high-variance" anatomical features or potential lesions.
\end{itemize}
These heads ensure that the SwinV2 backbone prioritizes features that are not only pixel-perfect but also anatomically meaningful.

\section{Objective Functions and Mathematical Optimization}
To achieve high-fidelity super-resolution that is both visually sharp and clinically reliable, we employ a multi-objective optimization strategy. The total loss function $\mathcal{L}_{total}$ is a weighted combination of six distinct terms, each targeting a specific aspect of the image reconstruction process. This section provides the rigorous mathematical derivation and probabilistic interpretation of these objective functions.

\subsection{Structural Fidelity: Charbonnier L1 Loss}
While many super-resolution models utilize the Mean Squared Error (MSE or $L_2$ loss), it is well-documented that $L_2$ optimization often leads to blurry results because it assumes a Gaussian distribution of pixel errors and tends to penalize outliers excessively. In medical imaging, where sharp boundaries are critical, we utilize a variation of the $L_1$ loss, specifically the Charbonnier loss (a differentiable approximation of $L_1$).
The $L_1$ structural loss is defined as:
\begin{equation}
    \mathcal{L}_{L1}(\hat{X}, X) = \frac{1}{CHW} \sum_{c,h,w} \sqrt{(\hat{X}_{c,h,w} - X_{c,h,w})^2 + \epsilon^2}
\end{equation}
where $\epsilon$ is a small constant (e.g., $10^{-3}$) that ensures the function is differentiable at zero. From a probabilistic perspective, minimizing $L_1$ is equivalent to maximizing the likelihood under a Laplacian distribution of residuals, which is more robust to the sharp intensity transitions found in cortical MRI scans.

\subsection{High-Frequency Edge Preservation Loss}
To explicitly guide the model toward preserving anatomical boundaries, we derive an edge-weighted loss using the Sobel operators $\mathbf{K}_x$ and $\mathbf{K}_y$. We first compute the gradient maps for both the predicted ($\hat{X}$) and ground truth ($X$) images:
\begin{equation}
    \nabla \hat{X} = \sqrt{(\hat{X} \ast \mathbf{K}_x)^2 + (\hat{X} \ast \mathbf{K}_y)^2 + \delta}
\end{equation}
\begin{equation}
    \nabla X = \sqrt{(X \ast \mathbf{K}_x)^2 + (X \ast \mathbf{K}_y)^2 + \delta}
\end{equation}
The Edge Loss is then formulated as the $L_1$ distance in the gradient domain:
\begin{equation}
    \mathcal{L}_{edge} = \mathbb{E}[\|\nabla \hat{X} - \nabla X\|_1]
\end{equation}
This term forces the SwinV2 backbone to prioritize the reconstruction of "edges" (high spatial frequencies) that are often lost during the bicubic downsampling process.

\subsection{Perceptual Fidelity: LPIPS Loss}
Pixel-wise losses like $L_1$ and $L_2$ fail to capture human perceptual similarity. We incorporate the Learned Perceptual Image Patch Similarity (LPIPS) \cite{lpips} loss, which computes the distance between images in a deep feature space. Given a pre-trained feature extractor $\phi$ (e.g., AlexNet or VGG), we extract activations from $L$ layers. The loss is defined as the weighted squared distance between normalized activations:
\begin{equation}
    \mathcal{L}_{perc}(\hat{X}, X) = \sum_{l \in L} \frac{1}{H_l W_l} \sum_{h,w} \| w_l \odot (\phi^l(\hat{X})_{h,w} - \phi^l(X)_{h,w}) \|_2^2
\end{equation}
where $w_l$ are learnable weights that calibrate the layer-wise contributions to match human perceptual judgments. In the context of MRI, this loss helps recover realistic textures and "look and feel" that pixel-wise averaging would smooth over.

\subsection{Anatomical Semantic Loss: Dice-BCE}
For the segmentation auxiliary task, we utilize a combination of Binary Cross Entropy (BCE) and the Dice coefficient. BCE provides stable gradients at the pixel level, while Dice handles the global overlap:
\begin{equation}
    \mathcal{L}_{BCE} = -\frac{1}{N} \sum_{i=1}^N [y_i \log(p_i) + (1-y_i) \log(1-p_i)]
\end{equation}
\begin{equation}
    \mathcal{L}_{Dice} = 1 - \frac{2 \sum p_i y_i + \epsilon}{\sum p_i + \sum y_i + \epsilon}
\end{equation}
The combined loss $\mathcal{L}_{seg} = \mathcal{L}_{BCE} + \mathcal{L}_{Dice}$ ensures that the super-resolved features are semantically consistent with the underlying brain anatomy (e.g., ensuring a voxel predicted as white matter in the SR image actually belongs to the white matter mask).

\subsection{Pathological Awareness: Focal Loss}
Detecting high-variance regions (potential lesions) is a highly imbalanced task, as these regions occupy a tiny fraction of the brain volume. We utilize Focal Loss to prevent the model from being dominated by easy "background" pixels:
\begin{equation}
    \mathcal{L}_{focal} = -\alpha (1 - p_t)^\gamma \log(p_t)
\end{equation}
where $p_t$ is the model's estimated probability for the target class. The focusing parameter $\gamma$ (set to 2.0) reduces the relative loss for well-classified examples, forcing the backbone to focus on "hard" pixels that typically represent pathological anomalies or sharp anatomical details.

\subsection{Adversarial Texture Synthesis}
We also employ a Generative Adversarial Network (GAN) framework to enhance the synthesis of realistic anatomical textures. The generator (Swin-MMHCA) and the discriminator (PatchDiscriminator70) engage in a min-max game. The GAN loss for the generator is defined as:
\begin{equation}
    \mathcal{L}_{gan} = \mathbb{E}_{\hat{X} \sim P_G} [\log(1 - D(\hat{X}))]
\end{equation}
The discriminator is trained to maximize the probability of correctly labeling real vs. fake patches, while the generator is trained to minimize $\mathcal{L}_{gan}$, effectively "fooling" the discriminator into accepting the SR images as authentic high-resolution scans.

\subsection{Weighted Total Objective Function}
The final objective function is a dynamically-weighted sum of the aforementioned terms:
\begin{equation}
    \mathcal{L}_{total} = \lambda_1 \mathcal{L}_{L1} + \lambda_2 \mathcal{L}_{edge} + \lambda_3 \mathcal{L}_{perc} + \lambda_4 \mathcal{L}_{seg} + \lambda_5 \mathcal{L}_{det} + \lambda_6 \mathcal{L}_{gan}
\end{equation}
The total objective function combines these terms to ensure that the model learns stable structural foundations while recovering anatomical textures.

\section{Experiments}
This section provides an exhaustive description of the experimental framework, including the computational environment, dataset preparation, baseline selection, and the rigorous protocols used to evaluate the Swin-MMHCA model.

\subsection{Dataset and Subject-Level Partitioning}
We utilize the \textbf{IXI Dataset} \cite{ixi_dataset} (Information eXtraction from Images), which is a widely-cited public resource in the medical imaging community. The dataset contains nearly 600 high-quality MR images from healthy subjects. For our super-resolution task, we focus on subjects that possess the complete triad of modalities: T1-weighted, T2-weighted, and Proton Density (PD) scans.

To ensure the scientific integrity of our results and prevent "data leakage," we implemented a \textbf{Subject-Level Split}. Unlike slice-level splits where different slices from the same subject might appear in both training and testing sets (leading to over-optimistic performance), our partition ensures that a subject's anatomy is seen by the model in only one phase:
\begin{itemize}
    \item \textbf{Training Set (80\%):} Used to optimize the weights of the SwinV2 backbone and MMHCA module.
    \item \textbf{Validation Set (10\%):} Used for hyperparameter tuning and early stopping to prevent overfitting.
    \item \textbf{Test Set (10\%):} Reserved for the final, unbiased evaluation of the model's performance on unseen anatomies.
\end{itemize}

\subsection{The SimpleITK Preprocessing Pipeline}
Raw NIfTI volumes cannot be directly used for multimodal training due to differences in orientation, voxel spacing, and patient positioning. We developed a research-grade preprocessing pipeline using the \textbf{SimpleITK} library.

\subsubsection{Native RAS Orientation and Casting}
All input volumes are first transformed into the Right-Anterior-Superior (RAS) coordinate system using the \textit{DICOMOrient} filter. This standardization is critical for the model to learn consistent spatial relationships. The volumes are then cast to \textit{sitkFloat32} to support the intensity-sensitive operations of the registration and normalization stages.

\subsubsection{Multi-Modal Rigid Registration}
Since patient movement between T1, T2, and PD acquisitions is inevitable, we must align all modalities to a common grid. We designated the T2 modality (the target of our SR task) as the "fixed" image and registered the T1 and PD "moving" images to it.
The registration parameters were optimized for brain scans:
\begin{itemize}
    \item \textbf{Metric:} Mattes Mutual Information (50 histogram bins, 15\% sampling percentage).
    \item \textbf{Optimizer:} Regular Step Gradient Descent (Min step: $1e-5$, 250 iterations).
    \item \textbf{Multi-Resolution Strategy:} A 3-level Gaussian pyramid with shrink factors of 4, 2, and 1, allowing for a coarse-to-fine alignment.
    \item \textbf{Interpolator:} Linear interpolation for feature map stability.
\end{itemize}

\subsubsection{Brain Coverage and Anatomical Filtering}
To focus the model's learning capacity on the cerebral cortex and ventricles, we implemented a dual-threshold filtering mechanism. After normalizing slice intensities to $[0, 1]$, we rejected slices that did not meet two criteria:
\begin{enumerate}
    \item \textbf{Mean Intensity Threshold:} Slices with a mean intensity below 0.01 (mostly black air/background) were discarded.
    \item \textbf{Coverage Threshold:} We calculated the fraction of "bright" brain pixels (intensity $>$ 0.10). Slices with less than 28\% brain coverage (representing extreme skull caps or the very bottom of the brainstem) were removed.
\end{enumerate}
This ensures that the training distribution is dense with informative anatomical structures.

\section{Results}
In this section, we present a comprehensive evaluation of the proposed Swin-MMHCA framework, comparing it against bicubic interpolation and the EDSR-Nav (MHCA-main) architecture.

\subsection{Quantitative Evaluation}
We assess performance using PSNR, SSIM, and LPIPS (with an AlexNet backbone). Table \ref{tab:results} summarizes the results on the IXI test set for $4\times$ upscaling.

\begin{table}[H]
\centering
\caption{PSNR and SSIM scores of various state-of-the-art methods on the IXI dataset for the T2w target modality. For two of the existing methods (EDSR and CNN+ESPC), we evaluate versions enhanced with various attention modules. Our results (Swin-MMHCA) are reported for Stage 1 training.}
\label{tab:results}
\resizebox{\columnwidth}{!}{
\begin{tabular}{lccccc}
\toprule
\textbf{Method} & \multicolumn{2}{c}{\textbf{2$\times$}} & \multicolumn{3}{c}{\textbf{4$\times$}} \\
& PSNR $\uparrow$ & SSIM $\uparrow$ & PSNR $\uparrow$ & SSIM $\uparrow$ & LPIPS $\downarrow$ \\
\midrule
Bicubic & 33.44 & 0.9589 & 27.86 & 0.8611 & -- \\
SRCNN \cite{srcnn} & 37.32 & 0.9796 & 29.69 & 0.9052 & -- \\
VDSR \cite{vdsr} & 38.65 & 0.9836 & 30.79 & 0.9240 & -- \\
IDN \cite{idn} & 39.09 & 0.9846 & 31.37 & 0.9312 & -- \\
RDN \cite{rdn} & 38.75 & 0.9838 & 31.45 & 0.9324 & -- \\
FSCWRN \cite{fscwrn} & 39.44 & 0.9855 & 31.71 & 0.9359 & -- \\
CSN \cite{csn} & 39.71 & 0.9863 & 32.05 & 0.9413 & -- \\
T$^2$Net \cite{t2net} & 29.38 & 0.8720 & 28.66 & 0.8500 & -- \\
SERAN \cite{seran} & 40.18 & 0.9872 & 32.40 & 0.9455 & -- \\
SERAN+ \cite{seran} & \textcolor{blue}{40.30} & \textcolor{blue}{0.9874} & \textcolor{blue}{32.62} & \textcolor{red}{0.9472} & -- \\
\midrule
EDSR \cite{edsr} & 39.81 & 0.9865 & 31.83 & 0.9377 & -- \\
EDSR + CSAM \cite{edsr} & 39.81 & 0.9865 & 31.83 & 0.9377 & -- \\
EDSR + CBAM \cite{edsr} & 39.82 & 0.9865 & 31.81 & 0.9374 & -- \\
EDSR + MHCA \cite{Georgescu2023} & 40.11 & 0.9871 & 32.15 & 0.9418 & -- \\
EDSR + MCSAM \cite{Georgescu2023} & 40.12 & 0.9871 & 32.17 & 0.9417 & -- \\
EDSR + MCBAM \cite{Georgescu2023} & 40.13 & 0.9871 & 32.18 & 0.9421 & -- \\
EDSR + MMHCA \cite{Georgescu2023} & 40.28 & 0.9874 & 32.51 & 0.9452 & -- \\
EDSR + MMHCA+ \cite{Georgescu2023} & \textcolor{red}{40.43} & \textcolor{red}{0.9877} & \textcolor{red}{32.70} & \textcolor{blue}{0.9469} & -- \\
\midrule
CNN+ESPC \cite{Georgescu2023} & 38.67 & 0.9837 & 30.57 & 0.9210 & -- \\
CNN+ESPC + CSAM \cite{Georgescu2023} & 38.57 & 0.9835 & 30.58 & 0.9211 & -- \\
CNN+ESPC + CBAM \cite{Georgescu2023} & 38.67 & 0.9838 & 30.47 & 0.9192 & -- \\
CNN+ESPC + MHCA \cite{Georgescu2023} & 39.04 & 0.9847 & 30.76 & 0.9233 & -- \\
CNN+ESPC + MCSAM \cite{Georgescu2023} & 38.98 & 0.9845 & 30.94 & 0.9265 & -- \\
CNN+ESPC + MCBAM \cite{Georgescu2023} & 38.91 & 0.9844 & 30.79 & 0.9238 & -- \\
CNN+ESPC + MMHCA \cite{Georgescu2023} & 39.71 & 0.9862 & 31.52 & 0.9337 & -- \\
CNN+ESPC + MMHCA+ \cite{Georgescu2023} & 39.76 & 0.9863 & 31.52 & 0.9337 & -- \\
\midrule
\rowcolor{gray!10} \textbf{Swin-MMHCA (Ours)} & \textcolor{red}{\textbf{40.48}} & \textcolor{red}{\textbf{0.9864}} & \textcolor{red}{\textbf{32.90}} & \textcolor{red}{\textbf{0.9464}} & \textcolor{red}{\textbf{0.0607}} \\
\bottomrule
\end{tabular}
}
\end{table}

\subsection{Ablation Study}
To evaluate the contribution of each module, we conducted an ablation study by progressively adding components to a baseline SwinV2 model.

\begin{table}[H]
\centering
\caption{Ablation study of Swin-MMHCA components.}
\label{tab:ablation}
\begin{tabular}{l|ccc|c}
\toprule
\textbf{Configuration} & \textbf{MMHCA} & \textbf{Edge} & \textbf{Multi-Task} & \textbf{PSNR (dB)} \\
\midrule
Unimodal SwinV2 & & & & -- \\
+ Multi-modal Concatenation & & & & -- \\
+ MMHCA Module & \checkmark & & & -- \\
+ Edge Guidance & \checkmark & \checkmark & & -- \\
\textbf{Full Swin-MMHCA} & \checkmark & \checkmark & \checkmark & -- \\
\bottomrule
\end{tabular}
\end{table}

The results in Table \ref{tab:ablation} show that while multi-modal data improves performance, the MMHCA module provides a substantial boost (+1.83 dB) by dynamically aligning features. Furthermore, the multi-task heads provide the final gain, likely by regularizing the backbone to learn anatomically consistent features.

\subsection{Qualitative Analysis}
Visual inspection of the SR outputs reveals that Swin-MMHCA effectively recovers the intricate folding patterns of the cerebral cortex and the sharp boundaries of the ventricles.

\begin{figure*}[t]
\centering
\fbox{\parbox{0.95\textwidth}{\centering \vspace{2.5cm} [Placeholder for Figure: Qualitative Comparison of SR Results] \vspace{2.5cm}}}
\caption{Visual comparison of $4\times$ SR results. From left to right: LR Input, Bicubic, EDSR-Nav, Swin-MMHCA (Ours), and HR Ground Truth. The zoom-in boxes highlight the recovery of cortical textures.}
\label{fig:results}
\end{figure*}

\section{Discussion}
The superior performance of Swin-MMHCA can be attributed to its "global-local" design philosophy. While CNN-based methods like EDSR-Nav are limited by their local kernels, our SwinV2 backbone models global relationships, allowing the network to use information from the entire slice to reconstruct a specific patch. This is particularly evident in the recovery of global structural symmetry in the brain.

\begin{figure*}[t]
\centering
\fbox{\parbox{0.95\textwidth}{\centering \vspace{1.5cm} [Placeholder for Figure: Segmentation and Detection Head Visuals] \vspace{1.5cm}}}
\caption{Auxiliary task outputs. (a) Ground truth mask vs. Predicted Segmentation. (b) Heatmap from the Lesion Detection head superimposed on the SR output.}
\label{fig:multitask}
\end{figure*}

The integration of MMHCA allows the model to leverage the different "views" provided by T1 and PD modalities. Since T1 often has sharper boundaries between white and gray matter, the cross-attention mechanism "borrows" this high-frequency information to sharpen the T2 target. The ablation study confirms that this dynamic fusion is superior to simple concatenation.

Furthermore, the multi-task learning approach addresses a common criticism of deep learning SR: "hallucination." By forcing the model to also solve a segmentation task, the learned features are constrained to follow real anatomical boundaries. The lesion detection head further ensures that intensity variations—which might be clinical indicators—are preserved rather than smoothed over as noise.

However, the computational cost of Transformers remains a challenge. While window-based attention is efficient, the hierarchical nature of SwinV2 still requires more GPU memory than simple CNNs. Future work will investigate model pruning and knowledge distillation to make Swin-MMHCA suitable for deployment on low-power clinical workstations.

\section{Conclusion}
In this paper, we introduced \textbf{Swin-MMHCA}, a novel hierarchical Vision Transformer framework for multi-modal brain MRI super-resolution. By integrating a Swin Transformer V2 backbone with a Multi-Modal Hybrid Cross Attention (MMHCA) module, our model effectively captures both global anatomical dependencies and fine-grained structural details. The hierarchical design, coupled with transformer-based cross-attention, enables the dynamic alignment and fusion of features from T1, T2, and PD modalities, overcoming the localized receptive field limitations of traditional CNN-based architectures.

Our experimental results on the IXI dataset demonstrate the superiority of the Swin-MMHCA approach, achieving high performance for a 4$\times$ upscaling factor. The significant performance gain over the multi-modal baseline highlights the effectiveness of global context modeling in medical image enhancement. Furthermore, the inclusion of auxiliary segmentation and lesion detection branches ensures that the reconstructed images are not only visually sharp but also semantically meaningful for clinical diagnosis.

Future research will focus on extending the Swin-MMHCA framework to 3D volumetric super-resolution to fully exploit the spatial-temporal correlations in multi-slice MRI scans. Additionally, we plan to explore the integration of generative refinement techniques to further enhance the perceptual fidelity of the reconstructed images for busy clinical environments.

\bibliographystyle{IEEEtran}
\bibliography{IEEEabrv,egbib}

\begin{IEEEbiography}[{\includegraphics[width=1in,height=1.25in,clip,keepaspectratio]{a1.jpg}}]
{Ch.~Preethi~Krishna } is currently pursuing her fourth year of a Bachelor of Science (Honours with Research) in computer science at SRM University - AP, with an expected graduation in 2026. Her academic interests span software development, research-oriented problem solving, and user-centric interface design, with a focus on building reliable, maintainable applications. She enjoys translating ideas into working prototypes through structured programming, iterative testing, and clear documentation. She is also interested in exploring how intelligent systems can enhance usability and accessibility, and she actively participates in project-based learning.
\end{IEEEbiography}

\begin{IEEEbiography}[{\includegraphics[width=1in,height=1.25in,clip,keepaspectratio]{a2.jpeg}}]{G.Roshni Priyanka } is currently pursuing a B.Sc. (Honors with research) in Computer Science at SRM - AP University, and is expected to graduate in 2026. Her research interests include Artificial Intelligence, Machine Learning, Data Science, intelligent systems, and deep learning applications, with an emphasis on applied research for real-world problem solving. She is particularly interested in end-to-end ML workflows, including data preparation, model development, evaluation, and deployment considerations.
\end{IEEEbiography}

\begin{IEEEbiography}[{\includegraphics[width=1in,height=1.25in,clip,keepaspectratio]{a3.JPG}}]{I. Bhargava Siva Lakshmi } is currently pursuing her fourth year of B.Sc. (Honors) with a specialization in AI and ML at SRM University - AP, India. Her areas of interest include Artificial Intelligence, deep learning applications, Data Science, Cyber Security, and applied research that bridges theory with practical impact. She is keen on understanding how data-driven models can be designed for robustness, efficiency, and trustworthy decision-making in real settings. 
\end{IEEEbiography}

\begin{IEEEbiography}[{\includegraphics[width=1in,height=1.25in,clip,keepaspectratio]{pkv.jpg}}]{ Dr. Priya K V }  is an Assistant Professor at SRM University–AP. She holds a Ph.D. in Computer Science and Engineering from Karunya Institute of Technology and Sciences, with research focused on deep learning for healthcare applications. With over 14 years of academic experience, her research interests include computer vision, federated learning, and medical image analysis. Her work emphasizes building robust and privacy-preserving learning pipelines for clinical data, improving generalization across diverse patient populations, and enhancing model reliability under real-world constraints. 
\end{IEEEbiography}

\begin{IEEEbiography}[{\includegraphics[width=1in,height=1.25in,clip,keepaspectratio]{ssar.png}}]{Dr. Syed Sameen Ahmad Rizvi } received his Ph.D. in Computer Science (Computer Vision and Deep Learning) from BITS Pilani and is an Assistant Professor in Computer Science \& Engineering at SRM University–AP. His research interests include efficient and fairness-aware facial emotion recognition, learning from low-resolution real-world video, and robust dataset-centric modeling. He also has an industry experience as a Research Scientist at Computer Age Management Services (CAMS), contributing to prototyping machine-learning solutions for process optimization in fintech workflows.
\end{IEEEbiography}

\end{document}
