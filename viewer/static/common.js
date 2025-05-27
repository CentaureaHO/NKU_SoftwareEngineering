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

// 设置提示框和提示灯
function updateAlertBox(message) {
    const textArea = document.querySelector('.alert-box textarea');
    if (textArea) { textArea.value = message; }
}

function fetchAndUpdateString() {
    fetch('/get_note')
        .then(response => response.json())
        .then(data => { updateAlertBox(data.updated_message); })
        .catch(error => { console.error('请求失败:', error); });
}

function get_light() {
    fetch('/get_light')
    .then(response => response.json())
    .then(data => {
        const alertLight = document.querySelector('.alert-light');
        const color = data.color;
        const blink = data.blink;

        // 清除所有现有状态并设置新状态
        alertLight.classList.remove('red', 'green', 'blinking');
        if (blink === true) { alertLight.classList.add('blinking'); }
        if (color === 'red') { alertLight.classList.add('red'); }
        else if (color === 'green') { alertLight.classList.add('green');  }
    })
    .catch(err => console.error('获取闪烁状态失败:', err));
}

// 页面跳转
function pollAction() {
    fetch('/get_action')
    .then(response => response.json())
    .then(data => {
        if (data.action) {
            console.log("🎯 检测到动作:", data.action);
            console.log("🚀 正在跳转到: /" + data.action);
            window.location.href = '/' + data.action;
        }
    })
    .catch(error => console.error('❌ 轮询错误:', error));
}

window.onload = () => {
    fetchAndUpdateString();
    get_light();
    setInterval(fetchAndUpdateString, 2000);
    setInterval(get_light, 2000);
    setInterval(pollAction, 2000);
};