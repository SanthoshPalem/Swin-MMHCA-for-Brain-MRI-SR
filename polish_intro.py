import re

file_path = 'ResearchPaper/ojim.tex'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

contributions_text = r'''The key contributions of this work include:
\begin{enumerate}
    \item A specialized Swin Transformer V2-based hierarchical architecture that captures both local textures and global anatomical context for brain MRI.
    \item The introduction of a Multi-Modal Hybrid Cross Attention (MMHCA) mechanism to effectively exploit inter-modal correlations for enhanced texture recovery from T1 and PD modalities.
    \item A multi-task learning strategy incorporating segmentation and lesion detection heads to enforce semantic consistency in the super-resolved outputs.
    \item A research-grade preprocessing pipeline using SimpleITK for multi-modal registration and brain coverage filtering.
    \item Extensive validation on the IXI dataset, demonstrating superior performance in terms of PSNR, SSIM, and LPIPS compared to existing CNN-based and multi-modal baselines.
\end{enumerate}

The remainder of this paper is organized as follows. Section II provides an exhaustive literature review and establishes the theoretical foundations. Section III details the proposed Swin-MMHCA architecture. Section IV describes the experimental setup and preprocessing pipeline. Section V presents the quantitative and qualitative results, followed by a discussion in Section VI. Finally, Section VII concludes the paper and outlines future directions.'''

# Find the contributions block that is currently floating
floating_contrib_pattern = re.compile(r'The key contributions of this work include:.*?\\end\{enumerate\}', re.DOTALL)
content_no_contrib = floating_contrib_pattern.sub('', content)

# Find the end of subsection 1.5 and insert the polished end-of-intro
intro_end_marker = "clinically faithful representation of the patient's anatomy."
if intro_end_marker in content_no_contrib:
    final_content = content_no_contrib.replace(intro_end_marker, intro_end_marker + "\n\n" + contributions_text)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(final_content)
    print('Introduction polished and contributions consolidated.')
else:
    print('Error: Intro end marker not found.')
