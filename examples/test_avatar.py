from core.plugins.lip_sync.core.avatar_extentions import AvatarManager, avatar_file_writer, avatar_file_writer2
from core.tools.file_tools import VideoWriter
import time

text = "Healthcare in the United States is a complex and multifaceted system that combines public and private elements, presenting both opportunities and challenges for individuals, policymakers, and the economy. It is often described as one of the most advanced healthcare systems in the world in terms of medical technology, research, and access to specialized care. However, it is also marked by disparities in access, affordability, and outcomes, which contribute to ongoing debates about reform and improvement. The healthcare system is largely dominated by private insurance companies, with a significant portion of the population receiving coverage through employer-sponsored insurance plans. In addition to these private plans, there are public health programs like Medicaid and Medicare that provide coverage for low-income individuals and those over the age of 65, respectively. However, a substantial number of Americans remain uninsured or underinsured, which has led to concerns about the overall effectiveness and equity of the system."
avatar_manager = AvatarManager()
buffer = avatar_manager.tts_buffer('mag', text)
for video, audio, text in buffer:
    avatar_file_writer("a.mp4", video, audio)
    break
