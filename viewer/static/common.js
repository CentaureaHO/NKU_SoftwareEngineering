// 隐藏跳转按钮
document.addEventListener('DOMContentLoaded', function() {
    const controlButtons = document.getElementById('control-buttons');
    const controlArea = document.querySelector('.controls'); // 父容器
    let hideTimeout;
    const hideDelay = 1000;
            
    // 初始状态 - 页面加载5秒后自动隐藏
    setTimeout(() => { controlButtons.classList.add('hidden'); }, 5000);

    // 监听鼠标移动
    controlArea.addEventListener('mousemove', function(e) {
        // 计算鼠标距离底部的距离
        const distanceFromBottom = controlArea.clientHeight - (e.pageY - controlArea.getBoundingClientRect().top);

        // 如果鼠标靠近底部，显示按钮
        if (distanceFromBottom < 150) { // 底部150px区域
            showButtons();
            clearTimeout(hideTimeout);
        }
        else {
            // 鼠标远离底部，延时隐藏
            clearTimeout(hideTimeout);
            hideTimeout = setTimeout(hideButtons, hideDelay);
        }
    });
            
    // 鼠标悬停在按钮上时保持显示
    controlButtons.addEventListener('mouseenter', function() {
        clearTimeout(hideTimeout);
        showButtons();
    });
            
    // 鼠标离开按钮时延时隐藏
    controlButtons.addEventListener('mouseleave', function() {
        hideTimeout = setTimeout(hideButtons, hideDelay);
    });
            
    // 离开控制区域时延时隐藏
    controlArea.addEventListener('mouseleave', function() {
        hideTimeout = setTimeout(hideButtons, hideDelay);
    });
            
    function showButtons() {
        controlButtons.classList.remove('hidden');
    }
            
    function hideButtons() {
        controlButtons.classList.add('hidden');
    }
});

// 开启前置摄像头
async function initFrontCamera() {
    try {
        // 请求摄像头访问权限，指定使用前置摄像头
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "user" },
            audio: false
        });
                
        // 获取视频元素并设置视频流
        const videoElement = document.getElementById('frontCamera');
        videoElement.srcObject = stream;
        await videoElement.play().catch(e => console.error('播放失败:', e));
        console.log('✅ 前置摄像头初始化成功');
        } 
    catch (error) {
        console.error('❌ 摄像头初始化失败:', error);
        // 显示错误信息
        const cameraFeed = document.querySelector('.camera-feed');
        cameraFeed.innerHTML = '<div class="camera-error">摄像头访问失败</div>';
                
        // 添加错误状态样式
        document.querySelector('.camera-status').style.color = 'red';
    }
}
        
// 页面加载完成后初始化摄像头
window.addEventListener('DOMContentLoaded', () => { initFrontCamera(); });