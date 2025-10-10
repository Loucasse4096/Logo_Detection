"""
Logo Detection Frontend - Simplified Version
Simple interface to create and monitor logo detection jobs
"""

import streamlit as st
import requests
import time
import tempfile
import base64
import json
import os
from datetime import datetime
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Logo Detection System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")

def check_api_health():
    """Check backend API health"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200 and response.json().get('status') == 'ok'
    except Exception as e:
        st.error(f"API Error: {e} - URL: {API_BASE_URL}")
        return False

def get_available_detectors():
    """Get list of available detectors"""
    try:
        response = requests.get(f"{API_BASE_URL}/models")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def upload_detection_job(video_file, logo_file, detector="orb"):
    """Create a new detection job"""
    try:
        files = {
            'video': video_file,
            'logo': logo_file
        }
        params = {'detector': detector}
        
        response = requests.post(
            f"{API_BASE_URL}/detect",
            files=files,
            params=params,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def get_job_status(job_id):
    """Get job status"""
    try:
        response = requests.get(f"{API_BASE_URL}/status/{job_id}")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def get_job_result(job_id):
    """Get job results"""
    try:
        response = requests.get(f"{API_BASE_URL}/result/{job_id}")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def download_result_video(job_id):
    """Download video with detections"""
    try:
        response = requests.get(f"{API_BASE_URL}/download/{job_id}")
        if response.status_code == 200:
            return response.content
        return None
    except:
        return None

def get_jobs_from_backend():
    """Get all jobs from backend"""
    try:
        response = requests.get(f"{API_BASE_URL}/jobs")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def clear_jobs_from_backend():
    """Clear job history on backend"""
    try:
        response = requests.delete(f"{API_BASE_URL}/jobs")
        if response.status_code == 200:
            return True
        return False
    except:
        return False

def delete_job_from_backend(job_id):
    """Delete a specific job from backend"""
    try:
        response = requests.delete(f"{API_BASE_URL}/jobs/{job_id}")
        if response.status_code == 200:
            return True, response.json().get('message', 'Job deleted successfully')
        return False, response.json().get('detail', 'Error deleting job')
    except Exception as e:
        return False, f"Error: {str(e)}"

def get_video_base64(video_content):
    """Convert video content to base64 for display"""
    if video_content:
        return base64.b64encode(video_content).decode('utf-8')
    return None

def display_video(video_content, title="Video"):
    """Display a video in Streamlit"""
    if video_content:
        try:
            st.markdown(f"**{title}**")
            # Use native Streamlit video player (works well with H.264 codec)
            st.video(video_content, format="video/mp4", start_time=0)
        except Exception as e:
            st.error(f"❌ Error displaying video: {e}")
            st.info("💡 You can still download the video using the Download button.")
    else:
        st.error("❌ No video available")

def save_temp_video(video_content, prefix="video"):
    """Temporarily save a video for display"""
    if video_content:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", prefix=prefix)
        temp_file.write(video_content)
        temp_file.close()
        return temp_file.name
    return None

def main():
    st.title("🎯 Logo Detection System")
    st.markdown("Logo detection in videos with multiple algorithms")

    # Initialize session state
    if 'selected_page_index' not in st.session_state:
        st.session_state.selected_page_index = 0
    
    # Check if we need to redirect to Jobs & History
    if st.session_state.get('redirect_to_jobs', False):
        st.session_state.selected_page_index = 1  # Jobs & History
        st.session_state.redirect_to_jobs = False
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🚀 New Job", "📊 Jobs & History"],
        index=st.session_state.selected_page_index
    )
    
    # Update selected index based on current selection
    pages = ["🚀 New Job", "📊 Jobs & History"]
    st.session_state.selected_page_index = pages.index(page)
    
    # API check
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔌 API Status")
    
    if check_api_health():
        st.sidebar.success("✅ API connected")
    else:
        st.sidebar.error("❌ API unavailable")
        st.sidebar.markdown("""
        **To start the API:**
        ```bash
        cd logo_detection_backend
        python run.py
        ```
        """)
        return
    
    # Page navigation
    if page == "🚀 New Job":
        create_job_page()
    elif page == "📊 Jobs & History":
        jobs_history_page()

def create_job_page():
    """Page to create a new detection job"""
    st.header("🚀 Create New Job")
    
    # Get available detectors
    detectors_info = get_available_detectors()
    if not detectors_info:
        st.error("❌ Unable to retrieve available detectors")
        return
    
    # Display available detectors
    st.subheader("🤖 Available Detectors")
    
    detectors = detectors_info.get('models', [])
    detector_options = {}
    
    for detector in detectors:
        detector_options[f"{detector['name']} - {detector['speed']}"] = detector['id']
    
    selected_detector_name = st.selectbox(
        "Choose a detector:",
        list(detector_options.keys()),
        index=0  # ORB by default
    )
    selected_detector = detector_options[selected_detector_name]
    
    # Information about selected detector
    selected_info = next(d for d in detectors if d['id'] == selected_detector)
    with st.expander(f"ℹ️ Information about {selected_info['name']}"):
        st.write(f"**Description:** {selected_info['description']}")
        st.write(f"**Speed:** {selected_info['speed']}")
        st.write(f"**Robust to:** {', '.join(selected_info['robust_to'])}")
    
    st.markdown("---")
    
    # File upload
    st.subheader("📁 File Upload")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📹 Video**")
        video_file = st.file_uploader(
            "Choose a video",
            type=['mp4', 'avi', 'mov'],
            help="Supported formats: MP4, AVI, MOV"
        )
        
        if video_file:
            st.success(f"✅ {video_file.name} ({video_file.size / 1024 / 1024:.1f} MB)")
    
    with col2:
        st.markdown("**🖼️ Logo**")
        logo_file = st.file_uploader(
            "Choose a logo",
            type=['jpg', 'jpeg', 'png'],
            help="Supported formats: JPG, PNG"
        )
        
        if logo_file:
            st.success(f"✅ {logo_file.name} ({logo_file.size / 1024:.1f} KB)")
    
    # Original video preview
    if video_file:
        st.markdown("---")
        st.subheader("📹 Original Video Preview")
        
        # Display uploaded video
        video_bytes = video_file.read()
        video_file.seek(0)  # Reset file pointer for processing
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            display_video(video_bytes, "Video to process")
        
        with col2:
            st.markdown("**📊 Information**")
            st.write(f"**Name:** {video_file.name}")
            st.write(f"**Size:** {video_file.size / 1024 / 1024:.1f} MB")
            st.write(f"**Type:** {video_file.type}")
    
    # Logo preview
    if logo_file:
        st.markdown("---")
        st.subheader("🖼️ Logo Preview")
        
        logo_bytes = logo_file.read()
        logo_file.seek(0)  # Reset file pointer
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(logo_bytes, caption="Logo to detect", width=300)
        
        with col2:
            st.markdown("**📊 Logo Information**")
            st.write(f"**Name:** {logo_file.name}")
            st.write(f"**Size:** {logo_file.size / 1024:.1f} KB")
            st.write(f"**Type:** {logo_file.type}")
            st.write(f"**Selected detector:** {selected_info['name']}")
            st.write(f"**Speed:** {selected_info['speed']}")
    
    # Processing button
    if video_file and logo_file:
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Prevent double submission
            if 'job_creating' not in st.session_state:
                st.session_state.job_creating = False
            
            if st.button("🚀 Start Detection", type="primary", disabled=st.session_state.job_creating):
                st.session_state.job_creating = True
                with st.spinner("Creating job..."):
                    # Create job
                    result = upload_detection_job(video_file, logo_file, selected_detector)
                    
                    if result:
                        job_id = result['job_id']
                        st.success(f"✅ Job created successfully!")
                        st.info(f"**Job ID:** `{job_id}`")
                        st.info("💾 Job is automatically saved in backend persistent history")
                        
                        # Temporarily store original video in session state to review later
                        if 'original_video' not in st.session_state:
                            st.session_state.original_video = {}
                        st.session_state.original_video[job_id] = {
                            'content': video_bytes,
                            'filename': video_file.name,
                            'size': video_file.size
                        }
                        
                        st.markdown("---")
                        st.markdown("### 📊 Job Tracking")
                        
                        # Real-time tracking
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        while True:
                            status = get_job_status(job_id)
                            if status:
                                current_status = status.get('status', 'unknown')
                                progress = status.get('progress', 0)
                                message = status.get('message', '')
                                
                                progress_bar.progress(progress)
                                status_text.text(f"Status: {current_status} - {message}")
                                
                                if current_status in ['completed', 'failed']:
                                    break
                            
                            time.sleep(2)
                        
                        # Final results
                        if current_status == 'completed':
                            st.session_state.job_creating = False  # Reset flag
                            st.success("🎉 Processing completed successfully!")
                            
                            # Get results
                            result_data = get_job_result(job_id)
                            if result_data:
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Total frames", result_data['total_frames'])
                                
                                with col2:
                                    st.metric("Frames with logo", result_data['frames_with_logo'])
                                
                                with col3:
                                    st.metric("Processing time", f"{result_data['processing_time']:.1f}s")
                            
                            # Redirect to Jobs & History page
                            st.markdown("---")
                            st.subheader("🎬 View Result Video")
                            
                            st.info("💡 Your video is ready! Go to the **Jobs & History** page to view and download the result.")
                            
                            # Button to redirect to Jobs & History
                            if st.button("📊 Go to Jobs & History", type="primary"):
                                st.session_state.redirect_to_jobs = True
                                st.rerun()
                        
                        elif current_status == 'failed':
                            st.session_state.job_creating = False  # Reset flag
                            st.error("❌ Processing failed")
                            error_msg = status.get('error', 'Unknown error')
                            st.error(f"**Error:** {error_msg}")
                    else:
                        st.session_state.job_creating = False  # Reset flag
                        st.error("❌ Error creating job")

def jobs_history_page():
    """Page to view job history"""
    st.header("📊 Jobs & History")
    
    # Load jobs from backend
    with st.spinner("Loading history..."):
        jobs_data = get_jobs_from_backend()
    
    if not jobs_data:
        st.error("❌ Unable to load job history from backend")
        return
    
    jobs = jobs_data.get('jobs', [])
    
    # Automatically fetch results for completed jobs
    for job in jobs:
        if job.get('status') == 'completed' and 'result' not in job:
            result = get_job_result(job['job_id'])
            if result:
                job['result'] = result
    
    if not jobs:
        st.info("📝 No jobs created yet. Create your first job in the 'New Job' tab.")
        return
    
    # General statistics
    st.subheader("📈 Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Jobs", jobs_data.get('total', 0))
    
    with col2:
        st.metric("Completed", jobs_data.get('completed', 0))
    
    with col3:
        st.metric("Failed", jobs_data.get('failed', 0))
    
    with col4:
        st.metric("Processing", jobs_data.get('processing', 0))
    
    st.markdown("---")
    
    # Job list
    st.subheader("📋 Job List")
    
    for i, job in enumerate(jobs):
        with st.expander(f"Job {job['job_id'][:8]}... - {job['status'].upper()}", expanded=False):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Job ID:** `{job['job_id']}`")
                st.write(f"**Detector:** {job['detector'].upper()}")
                st.write(f"**Video:** {job.get('video_file', 'N/A')}")
                st.write(f"**Logo:** {job.get('logo_file', 'N/A')}")
                st.write(f"**Created:** {job.get('created_at', 'N/A')}")
            
            with col2:
                # Button to refresh status
                if st.button(f"🔄 Refresh", key=f"refresh_{i}"):
                    status = get_job_status(job['job_id'])
                    if status:
                        job['status'] = status.get('status', 'unknown')
                        job['progress'] = status.get('progress', 0)
                        job['message'] = status.get('message', '')
                        
                        if status.get('status') == 'completed':
                            # Get results
                            result = get_job_result(job['job_id'])
                            if result:
                                job['result'] = result
                        
                        st.rerun()
                
                # Status display
                status = job.get('status', 'unknown')
                if status == 'completed':
                    st.success("✅ Completed")
                elif status == 'failed':
                    st.error("❌ Failed")
                elif status in ['pending', 'processing']:
                    st.warning("⏳ Processing")
                else:
                    st.info(f"📊 {status}")
                
                # Delete button for all jobs
                if st.button(f"🗑️ Delete Job", key=f"delete_job_{i}", type="secondary"):
                    # Confirmation dialog
                    if st.session_state.get(f"confirm_delete_job_{i}", False):
                        # Actually delete the job
                        success, message = delete_job_from_backend(job['job_id'])
                        if success:
                            st.success(f"✅ {message}")
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")
                    else:
                        # Show confirmation
                        st.session_state[f"confirm_delete_job_{i}"] = True
                        st.warning("⚠️ Click again to confirm deletion")
                        st.rerun()
                
                # Display results if available
                if 'result' in job:
                    result = job['result']
                    st.write(f"**Frames:** {result['total_frames']}")
                    st.write(f"**Detections:** {result['frames_with_logo']}")
                    st.write(f"**Time:** {result['processing_time']:.1f}s")
                    
                    # Action buttons
                    col_btn1, col_btn2, col_btn3 = st.columns(3)
                    
                    with col_btn1:
                        # View result video
                        if st.button(f"📺 View Result", key=f"view_result_{i}"):
                            if 'show_video' not in st.session_state:
                                st.session_state.show_video = set()
                            
                            job_id = job['job_id']
                            if job_id in st.session_state.show_video:
                                st.session_state.show_video.remove(job_id)
                            else:
                                st.session_state.show_video.add(job_id)
                            st.rerun()
                    
                    with col_btn2:
                        # Download button with caching
                        video_content = download_result_video(job['job_id'])
                        if video_content:
                            st.download_button(
                                label="📥 Download",
                                data=video_content,
                                file_name=f"logo_detection_{job['job_id'][:8]}.mp4",
                                mime="video/mp4",
                                key=f"download_{i}"
                            )
                        else:
                            st.button("📥 Download", key=f"download_disabled_{i}", disabled=True)
                    
                    with col_btn3:
                        # Delete button
                        if st.button(f"🗑️ Delete", key=f"delete_{i}", type="secondary"):
                            # Confirmation dialog
                            if st.session_state.get(f"confirm_delete_{i}", False):
                                # Actually delete the job
                                success, message = delete_job_from_backend(job['job_id'])
                                if success:
                                    st.success(f"✅ {message}")
                                    # Clean up session state
                                    if 'show_video' in st.session_state and job['job_id'] in st.session_state.show_video:
                                        st.session_state.show_video.remove(job['job_id'])
                                    st.rerun()
                                else:
                                    st.error(f"❌ {message}")
                            else:
                                # Show confirmation
                                st.session_state[f"confirm_delete_{i}"] = True
                                st.warning("⚠️ Click again to confirm deletion")
                                st.rerun()
                    
                    # Show video if toggled
                    if st.session_state.get('show_video') and job['job_id'] in st.session_state.show_video:
                        st.markdown("---")
                        with st.spinner("Loading result video..."):
                            video_content = download_result_video(job['job_id'])
                            if video_content:
                                st.markdown(f"**🎬 Result Video - {job['detector'].upper()}**")
                                display_video(video_content, f"Job {job['job_id'][:8]}")
                            else:
                                st.error("❌ Unable to load result video")
    
    # Button to clean history
    st.markdown("---")
    if st.button("🗑️ Clear History", type="secondary"):
        if clear_jobs_from_backend():
            st.success("✅ History deleted successfully")
            st.rerun()
        else:
            st.error("❌ Error deleting history")

if __name__ == "__main__":
    main()