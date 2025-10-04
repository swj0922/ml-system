// API基础URL
const API_BASE = '';

// 页面加载时检查API状态并获取特征列表
document.addEventListener('DOMContentLoaded', function() {
    checkApiStatus();
    getFeatures();
    initializeModelSelector();
});

// 初始化模型选择器
function initializeModelSelector() {
    const modelSelect = document.getElementById('llm-model-select');
    const modelDescriptionText = document.getElementById('model-description-text');
    
    if (modelSelect && modelDescriptionText) {
        // 模型描述映射
        const modelDescriptions = {
            'gemini': 'Gemini 2.5 Flash: Google的快速响应模型，适合实时分析，支持流式输出',
            'qwen': 'Qwen: 阿里云的中文优化模型，深度理解中文语境，支持流式输出'
        };
        
        // 监听模型选择变化
        modelSelect.addEventListener('change', function() {
            const selectedModel = this.value;
            modelDescriptionText.textContent = modelDescriptions[selectedModel] || '未知模型';
        });
        
        // 初始化描述
        modelDescriptionText.textContent = modelDescriptions[modelSelect.value] || '未知模型';
    }
}

// 检查API状态
async function checkApiStatus() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();
        document.getElementById('api-status').textContent = data.status === 'healthy' ? '正常运行' : '异常';
        document.getElementById('api-status').style.color = data.status === 'healthy' ? '#2ecc71' : '#e74c3c';
    } catch (error) {
        document.getElementById('api-status').textContent = '无法连接';
        document.getElementById('api-status').style.color = '#e74c3c';
    }
}

// 获取特征列表
let featureNames = [];
async function getFeatures() {
    try {
        const response = await fetch(`${API_BASE}/features`);
        const data = await response.json();
        featureNames = data.features;
        createFeatureInputs();
        createShapStatsFeatureInputs();
    } catch (error) {
        showError('获取特征列表失败: ' + error.message);
    }
}

// 创建特征输入表单
function createFeatureInputs() {
    const container = document.getElementById('feature-inputs');
    container.innerHTML = '';
    
    featureNames.forEach(feature => {
        const inputGroup = document.createElement('div');
        inputGroup.className = 'input-group';
        
        const label = document.createElement('label');
        label.textContent = feature;
        
        const input = document.createElement('input');
        input.type = 'number';
        input.step = 'any';
        input.id = `feature-${feature}`;
        input.placeholder = `输入 ${feature} 的值`;
        
        inputGroup.appendChild(label);
        inputGroup.appendChild(input);
        container.appendChild(inputGroup);
    });
}



// 创建SHAP与统计量分析特征输入表单
function createShapStatsFeatureInputs() {
    const container = document.getElementById('shap-stats-feature-inputs');
    container.innerHTML = '';
    
    featureNames.forEach(feature => {
        const inputGroup = document.createElement('div');
        inputGroup.className = 'input-group';
        
        const label = document.createElement('label');
        label.textContent = feature;
        
        const input = document.createElement('input');
        input.type = 'number';
        input.step = 'any';
        input.id = `shap-stats-feature-${feature}`;
        input.placeholder = `输入 ${feature} 的值`;
        
        inputGroup.appendChild(label);
        inputGroup.appendChild(input);
        container.appendChild(inputGroup);
    });
}

// 获取随机样本
async function getRandomSample() {
    const count = document.getElementById('sample-count').value;
    
    try {
        showLoading(true);
        const response = await fetch(`${API_BASE}/sample?count=${count}`);
        const data = await response.json();
        
        displayRandomSample(data.samples);
    } catch (error) {
        showError('获取随机样本失败: ' + error.message);
    } finally {
        showLoading(false);
    }
}

// 显示随机样本
function displayRandomSample(samples) {
    // 存储样本数据到全局变量
    globalSampleData = samples;
    
    const container = document.getElementById('random-sample-data');
    container.innerHTML = '<h3>随机样本数据:</h3>';
    
    samples.forEach((sample, index) => {
        const sampleDiv = document.createElement('div');
        sampleDiv.className = 'result-container';
        
        let html = `<h4>样本 ${index + 1}</h4>`;
        html += '<table style="width: 100%; border-collapse: collapse;">';
        
        // 只显示前10个特征，避免表格过长
        const featuresToShow = Object.keys(sample).slice(0, 10);
        featuresToShow.forEach(feature => {
            if (feature !== 'Bankrupt') {  // 排除目标变量
                html += `<tr><td style="padding: 5px; border: 1px solid #ddd;">${feature}</td><td style="padding: 5px; border: 1px solid #ddd;">${sample[feature]}</td></tr>`;
            }
        });
        
        if (Object.keys(sample).length > 11) {  // 如果有超过10个特征
            html += `<tr><td colspan="2" style="padding: 5px; border: 1px solid #ddd; text-align: center;">... 还有 ${Object.keys(sample).length - 11} 个特征</td></tr>`;
        }
        
        html += '</table>';
        
        // 添加"用于PDP分析"按钮
        html += `<button onclick="useSampleForPDP(${index})" class="btn" style="margin-top: 10px;">用于PDP分析</button>`;
        
        sampleDiv.innerHTML = html;
        container.appendChild(sampleDiv);
    });
}

// 预测随机样本
async function predictRandomSample() {
    const count = document.getElementById('sample-count').value;
    
    try {
        showLoading(true);
        // 发送HTTP请求到后端API
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                sample_count: parseInt(count)
            })
        });
        
        const data = await response.json();
        displayPredictionResults(data);
    } catch (error) {
        showError('预测失败: ' + error.message);
    } finally {
        showLoading(false);
    }
}

// 预测手动输入
async function predictManualInput() {
    const features = {};
    
    featureNames.forEach(feature => {
        const input = document.getElementById(`feature-${feature}`);
        const value = parseFloat(input.value);
        
        if (isNaN(value)) {
            throw new Error(`请输入有效的 ${feature} 值`);
        }
        
        features[feature] = value;
    });
    
    try {
        showLoading(true);
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                features: [features]
            })
        });
        
        const data = await response.json();
        displayPredictionResults(data);
    } catch (error) {
        showError('预测失败: ' + error.message);
    } finally {
        showLoading(false);
    }
}

// 显示基本预测结果（不包含SHAP分析）
function displayPredictionResults(data) {
    const container = document.getElementById('prediction-results');
    container.innerHTML = '<h2>预测结果</h2>';
    
    data.predictions.forEach((prediction, index) => {
        const resultDiv = document.createElement('div');
        resultDiv.className = 'result-container';
        
        const probability = data.probabilities[index];
        const predictionText = prediction === 1 ? '破产' : '正常';
        const predictionColor = prediction === 1 ? '#e74c3c' : '#2ecc71';
        
        let html = `
            <div class="prediction-result">
                <h3>样本 ${index + 1}</h3>
                <p><strong>预测结果:</strong> <span style="color: ${predictionColor}">${predictionText}</span></p>
                <p><strong>破产概率:</strong> ${(probability * 100).toFixed(2)}%</p>
            </div>
        `;
        
        resultDiv.innerHTML = html;
        container.appendChild(resultDiv);
    });
}


// 切换标签页
function openTab(evt, tabName) {
    const tabcontent = document.getElementsByClassName("tabcontent");
    // 先隐藏所有内容
    for (let i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
    }
    // 移除所有标签的选中（active）状态
    const tablinks = document.getElementsByClassName("tablinks");
    for (let i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
    }
    // 再显示选中的内容
    document.getElementById(tabName).style.display = "block";
    evt.currentTarget.className += " active";
}

// 显示/隐藏加载状态
function showLoading(show) {
    document.getElementById('loading').style.display = show ? 'block' : 'none';
}

// 显示错误信息
function showError(message) {
    const errorElement = document.getElementById('error-message');
    errorElement.textContent = message;
    errorElement.style.display = 'block';
    
    // 5秒后自动隐藏错误信息
    setTimeout(() => {
        errorElement.style.display = 'none';
    }, 5000);
}


// 获取SHAP与统计量分析的随机样本
async function getRandomSampleForShapStats() {
    const count = document.getElementById('shap-stats-sample-count').value;
    
    try {
        showLoading(true);
        const response = await fetch(`${API_BASE}/sample?count=${count}`);
        const data = await response.json();
        
        displayShapStatsRandomSample(data.samples);
    } catch (error) {
        showError('获取随机样本失败: ' + error.message);
    } finally {
        showLoading(false);
    }
}

// 显示SHAP与统计量分析的随机样本
function displayShapStatsRandomSample(samples) {
    // 存储样本数据到全局变量
    globalSampleData = samples;
    
    const container = document.getElementById('shap-stats-random-sample-data');
    container.innerHTML = '<h3>随机样本数据:</h3>';
    
    samples.forEach((sample, index) => {
        const sampleDiv = document.createElement('div');
        sampleDiv.className = 'result-container';
        
        let html = `<h4>样本 ${index + 1}</h4>`;
        html += '<table style="width: 100%; border-collapse: collapse;">';
        
        // 只显示前10个特征，避免表格过长
        const featuresToShow = Object.keys(sample).slice(0, 10);
        featuresToShow.forEach(feature => {
            if (feature !== 'Bankrupt') {  // 排除目标变量
                html += `<tr><td style="padding: 5px; border: 1px solid #ddd;">${feature}</td><td style="padding: 5px; border: 1px solid #ddd;">${sample[feature]}</td></tr>`;
            }
        });
        
        if (Object.keys(sample).length > 11) {  // 如果有超过10个特征
            html += `<tr><td colspan="2" style="padding: 5px; border: 1px solid #ddd; text-align: center;">... 还有 ${Object.keys(sample).length - 11} 个特征</td></tr>`;
        }
        
        html += '</table>';
        
        // 添加"用于PDP分析"按钮
        html += `<button onclick="useSampleForPDP(${index})" class="btn" style="margin-top: 10px;">用于PDP分析</button>`;
        
        sampleDiv.innerHTML = html;
        container.appendChild(sampleDiv);
    });
}

// 分析随机样本（带统计量）
async function analyzeRandomSampleWithStats() {
    const count = document.getElementById('shap-stats-sample-count').value;
    
    try {
        showLoading(true);
        await getShapWithStatsAnalysis(count);
    } catch (error) {
        showError('SHAP与统计量分析失败: ' + error.message);
    } finally {
        showLoading(false);
    }
}

// 分析手动输入（带统计量）
async function analyzeManualInputWithStats() {
    const features = {};
    
    featureNames.forEach(feature => {
        const input = document.getElementById(`shap-stats-feature-${feature}`);
        const value = parseFloat(input.value);
        
        if (isNaN(value)) {
            throw new Error(`请输入有效的 ${feature} 值`);
        }
        
        features[feature] = value;
    });
    
    try {
        showLoading(true);
        await getShapWithStatsAnalysis(1, [features]);
    } catch (error) {
        showError('SHAP与统计量分析失败: ' + error.message);
    } finally {
        showLoading(false);
    }
}

// 获取SHAP与统计量分析
async function getShapWithStatsAnalysis(sampleCount, features = null) {
    try {
        showLoading(true);
        
        // 获取选择的模型
        const modelSelect = document.getElementById('llm-model-select');
        const selectedModel = modelSelect ? modelSelect.value : 'gemini';
        
        let requestBody;
        if (features) {
            requestBody = { 
                features: features,
                model: selectedModel
            };
        } else {
            requestBody = { 
                sample_count: parseInt(sampleCount),
                model: selectedModel
            };
        }
        
        console.log('选择的模型:', selectedModel);
        
        // 使用WebSocket进行流式分析
        await performStreamingShapAnalysis(requestBody);
    } catch (error) {
        showError('SHAP与统计量分析失败: ' + error.message);
    } finally {
        showLoading(false);
    }
}

// 使用WebSocket进行流式SHAP分析
async function performStreamingShapAnalysis(requestBody) {
    return new Promise((resolve, reject) => {
        // 构建WebSocket URL
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsHost = window.location.hostname;
        const wsPort = window.location.port || (window.location.protocol === 'https:' ? '443' : '80');
        const wsUrl = `${wsProtocol}//${wsHost}:${wsPort}/ws/shap-with-stats-analysis`;
        
        // 根据URL创建WebSocket连接
        const ws = new WebSocket(wsUrl);

        // 初始化变量
        let shapData = null;
        let llmInterpretationDiv = null;
        let llmMarkdownContent = ''; // 累积LLM输出的Markdown内容
        
        ws.onopen = function() {
            console.log('WebSocket连接已建立');
            // 向服务器发送请求数据
            ws.send(JSON.stringify(requestBody));
        };
        
        // 当WebSocket连接接收到来自服务器的消息时，这个函数会被自动调用
        // event是从服务器发来的数据
        ws.onmessage = function(event) {
            try {
                const message = JSON.parse(event.data);
                console.log('收到WebSocket消息:', message.type, message);
                
                if (message.type === 'status') {
                    // 更新前端状态显示
                    console.log('状态更新:', message.message, '时间:', new Date().toLocaleTimeString());
                    const loadingElement = document.querySelector('.loading');
                    if (loadingElement) {
                        loadingElement.textContent = message.message;
                    }
                    
                } else if (message.type === 'shap_results') {
                    // 接收到SHAP数据，异步显示基础结果
                    console.log('接收到SHAP结果:', '时间:', new Date().toLocaleTimeString());
                    shapData = message.data;
                    
                    // 更新状态显示
                    const loadingElement = document.querySelector('.loading');
                    if (loadingElement) {
                        loadingElement.textContent = '正在渲染SHAP结果...';
                    }
                    
                    // 异步处理SHAP结果显示，避免阻塞后续消息
                    setTimeout(() => {
                        // 显示SHAP分析结果
                        const container = document.getElementById('prediction-results');
                        container.innerHTML = '<h2>SHAP与统计量分析结果</h2>';
                        
                        shapData.shap_values.forEach((shapValues, index) => {
                            const resultDiv = document.createElement('div');
                            resultDiv.className = 'result-container';
                            
                            let html = `
                                <div class="prediction-result">
                                    <h3>样本 ${index + 1} 的SHAP分析</h3>
                                    <p><strong>基准值:</strong> ${shapData.base_value.toFixed(4)}</p>
                                </div>
                            `;
                            
                            // 添加SHAP解释
                            html += '<div class="feature-importance">';
                            html += '<h4>特征重要性 (SHAP值)</h4>';
                            
                            // 计算每个特征的SHAP值
                            const featureImportance = [];
                            
                            shapData.feature_names.forEach((name, i) => {
                                featureImportance.push({
                                    name: name,
                                    value: shapValues[i]
                                });
                            });
                            
                            // 按SHAP值绝对值排序
                            featureImportance.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
                            
                            // 显示前10个最重要的特征
                            featureImportance.slice(0, 10).forEach(feature => {
                                const isPositive = feature.value >= 0;
                                const barWidth = Math.min(Math.abs(feature.value) * 100, 300); // 限制最大宽度
                                
                                html += `
                                    <div class="feature-bar">
                                        <span class="feature-name">${feature.name}:</span>
                                        <span class="shap-bar ${isPositive ? 'positive' : 'negative'}" style="width: ${barWidth}px;"></span>
                                        <span class="shap-value">${feature.value.toFixed(4)}</span>
                                    </div>
                                `;
                            });
                            
                            html += '</div>';
                            resultDiv.innerHTML = html;
                            container.appendChild(resultDiv);
                        });
                        
                        // 准备LLM解读容器
                        const llmDiv = document.createElement('div');
                        llmDiv.className = 'result-container';
                        llmDiv.innerHTML = `
                            <div class="prediction-result">
                                <h3>LLM解读</h3>
                                <div id="llm-interpretation" class="llm-markdown-content" style="border: 1px solid #ddd; padding: 15px; background-color: #f9f9f9; min-height: 100px; line-height: 1.6;"></div>
                            </div>
                        `;
                        container.appendChild(llmDiv);
                        llmInterpretationDiv = document.getElementById('llm-interpretation');
                    }, 10); // 很短的延迟，让其他消息有机会被处理
                    
                } else if (message.type === 'llm_start') {
                    // LLM开始解读
                    console.log('LLM开始解读:', message.message, '时间:', new Date().toLocaleTimeString());
                    // 更新加载状态显示
                    const loadingElement = document.querySelector('.loading');
                    if (loadingElement) {
                        loadingElement.textContent = message.message;
                    }
                    
                } else if (message.type === 'llm_chunk') {
                    // 接收到LLM流式数据块
                    if (llmInterpretationDiv && message.content) {
                        // 累积Markdown内容
                        llmMarkdownContent += message.content;
                        
                        // 解析Markdown并渲染为HTML
                        try {
                            const htmlContent = marked.parse(llmMarkdownContent);
                            // 实时显示LLM输出的内容
                            llmInterpretationDiv.innerHTML = htmlContent;
                        } catch (error) {
                            // 如果Markdown解析失败，回退到纯文本显示
                            console.warn('Markdown解析失败，使用纯文本显示:', error);
                            llmInterpretationDiv.textContent = llmMarkdownContent;
                        }
                        
                        // 自动滚动到底部
                        llmInterpretationDiv.scrollTop = llmInterpretationDiv.scrollHeight;
                        
                        // 在控制台同步显示，与后端输出保持一致
                        console.log(message.content);
                    }
                    
                } else if (message.type === 'llm_complete') {
                    // LLM解读完成
                    console.log('LLM解读完成');
                    ws.close();
                    // 调用 resolve() 告诉 Promise 该异步操作已成功完成
                    resolve();
                    
                } else if (message.type === 'error') {
                    // 发生错误
                    console.error('WebSocket错误:', message.message);
                    reject(new Error(message.message));
                    ws.close();
                }
            } catch (error) {
                console.error('解析WebSocket消息失败:', error);
                reject(error);
            }
        };
        
        ws.onerror = function(error) {
            console.error('WebSocket连接错误:', error);
            reject(new Error('WebSocket连接失败'));
        };
        
        ws.onclose = function() {
            console.log('WebSocket连接已关闭');
        };
    });
}

// ==================== PDP功能相关函数 ====================

// PDP相关变量
let featureImportance = {};
let currentPDPSample = null;

// 全局样本数据存储，用于跨标签页共享
let globalSampleData = null;

// 使用选中的样本进行PDP分析
function useSampleForPDP(sampleIndex) {
    if (!globalSampleData || sampleIndex >= globalSampleData.length) {
        showError('样本数据不可用');
        return;
    }
    
    const sample = globalSampleData[sampleIndex];
    // 移除目标变量
    const sampleFeatures = { ...sample };
    delete sampleFeatures.Bankrupt;
    
    // 切换到PDP分析标签页
    const pdpTab = document.querySelector('[onclick="openTab(event, \'pdp-analysis\')"]');
    if (pdpTab) {
        pdpTab.click();
    }
    
    // 设置当前PDP样本
    currentPDPSample = sampleFeatures;
    
    // 显示样本数据
    displayPDPSample(sampleFeatures);
    
    // 显示提示信息
    const infoDiv = document.createElement('div');
    infoDiv.className = 'info-box';
    infoDiv.innerHTML = `<p><strong>已选择样本 ${sampleIndex + 1} 用于PDP分析</strong></p><p>该样本的特征值将在PDP图中以红色标注显示。</p>`;
    
    const container = document.getElementById('pdp-sample-data');
    container.insertBefore(infoDiv, container.firstChild);
    
    // 3秒后移除提示信息
    setTimeout(() => {
        if (infoDiv.parentNode) {
            infoDiv.parentNode.removeChild(infoDiv);
        }
    }, 3000);
}



// 页面加载时初始化PDP功能
document.addEventListener('DOMContentLoaded', function() {
    // 在现有的初始化函数中添加PDP初始化
    setTimeout(() => {
        initializePDPFeatures();
        getFeatureImportance();
        
        // 添加多进程选项的事件监听器
        const useMultiprocessingCheckbox = document.getElementById('use-multiprocessing');
        const processCountGroup = document.getElementById('process-count-group');
        
        if (useMultiprocessingCheckbox && processCountGroup) {
            useMultiprocessingCheckbox.addEventListener('change', function() {
                if (this.checked) {
                    processCountGroup.style.display = 'block';
                } else {
                    processCountGroup.style.display = 'none';
                }
            });
        }
    }, 1000); // 等待特征列表加载完成
});

// 初始化PDP特征选择列表
async function initializePDPFeatures() {
    if (featureNames.length === 0) {
        setTimeout(initializePDPFeatures, 500);
        return;
    }
    
    const container = document.getElementById('feature-selection-list');
    container.innerHTML = '';
    
    featureNames.forEach((feature, index) => {
        const featureItem = document.createElement('div');
        featureItem.className = 'feature-item';
        
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = `pdp-feature-${index}`;
        checkbox.value = feature;
        checkbox.addEventListener('change', updateFeatureSelection);
        
        const label = document.createElement('label');
        label.htmlFor = `pdp-feature-${index}`;
        label.textContent = feature;
        
        featureItem.appendChild(checkbox);
        featureItem.appendChild(label);
        container.appendChild(featureItem);
    });
}

// 获取特征重要性
async function getFeatureImportance() {
    try {
        const response = await fetch(`${API_BASE}/feature-importance`);
        const data = await response.json();
        featureImportance = data.feature_importance;
    } catch (error) {
        console.error('获取特征重要性失败:', error);
    }
}

// 更新特征选择状态
function updateFeatureSelection() {
    const checkboxes = document.querySelectorAll('#feature-selection-list input[type="checkbox"]');
    const selectedCount = Array.from(checkboxes).filter(cb => cb.checked).length;
    
    // 更新选择计数
    document.getElementById('selected-feature-count').textContent = `已选择: ${selectedCount}/9`;
    
    // 如果选择超过9个，禁用其他复选框
    if (selectedCount >= 9) {
        checkboxes.forEach(cb => {
            if (!cb.checked) {
                cb.disabled = true;
            }
        });
    } else {
        checkboxes.forEach(cb => {
            cb.disabled = false;
        });
    }
    
    // 更新选中项的样式
    checkboxes.forEach(cb => {
        const featureItem = cb.closest('.feature-item');
        if (cb.checked) {
            featureItem.classList.add('selected');
        } else {
            featureItem.classList.remove('selected');
        }
    });
}

// 随机选择特征
function selectRandomFeatures() {
    // 清空当前选择
    clearFeatureSelection();
    
    // 获取所有可用特征
    const checkboxes = document.querySelectorAll('#feature-selection-list input[type="checkbox"]');
    const allFeatures = Array.from(checkboxes).map(cb => cb.value);
    
    // 随机打乱特征数组并选择前9个
    const shuffledFeatures = allFeatures.sort(() => Math.random() - 0.5);
    const selectedFeatures = shuffledFeatures.slice(0, 9);
    
    // 勾选随机选择的特征
    checkboxes.forEach(cb => {
        if (selectedFeatures.includes(cb.value)) {
            cb.checked = true;
        }
    });
    
    updateFeatureSelection();
}

// 清空特征选择
function clearFeatureSelection() {
    const checkboxes = document.querySelectorAll('#feature-selection-list input[type="checkbox"]');
    checkboxes.forEach(cb => {
        cb.checked = false;
        cb.disabled = false;
    });
    updateFeatureSelection();
}

// 获取选中的特征
function getSelectedFeatures() {
    const checkboxes = document.querySelectorAll('#feature-selection-list input[type="checkbox"]:checked');
    return Array.from(checkboxes).map(cb => cb.value);
}

// 获取随机样本用于PDP分析
async function getRandomSampleForPDP() {
    try {
        showLoading(true);
        const response = await fetch(`${API_BASE}/sample?count=1`);
        const data = await response.json();
        
        if (data.samples && data.samples.length > 0) {
            currentPDPSample = data.samples[0];
            displayPDPSample(currentPDPSample);
        }
    } catch (error) {
        showError('获取随机样本失败: ' + error.message);
    } finally {
        showLoading(false);
    }
}

// 显示PDP样本数据
function displayPDPSample(sample) {
    const container = document.getElementById('pdp-sample-data');
    
    const html = `
        <h4>当前样本数据</h4>
        <div class="sample-highlight">
            <h5>样本特征值</h5>
            <p>该样本的特征值将在PDP图中以高亮点标注</p>
            <div style="max-height: 200px; overflow-y: auto; font-size: 12px;">
                ${Object.entries(sample).map(([key, value]) => 
                    `<div><strong>${key}:</strong> ${typeof value === 'number' ? value.toFixed(6) : value}</div>`
                ).join('')}
            </div>
        </div>
    `;
    
    container.innerHTML = html;
}

// 生成部分依赖图
async function generatePDP() {
    const selectedFeatures = getSelectedFeatures();
    
    if (selectedFeatures.length === 0) {
        showError('请至少选择一个特征');
        return;
    }
    
    if (selectedFeatures.length > 9) {
        showError('最多只能选择9个特征');
        return;
    }
    
    try {
        showLoading(true);
        
        const gridResolution = parseInt(document.getElementById('pdp-grid-resolution').value) || 100;
        const useMultiprocessing = document.getElementById('use-multiprocessing').checked;
        const processCount = parseInt(document.getElementById('process-count').value) || null;
        
        const requestBody = {
            features: selectedFeatures,
            sample_data: currentPDPSample,
            grid_resolution: gridResolution,
            use_multiprocessing: useMultiprocessing,
            n_processes: useMultiprocessing ? processCount : null
        };
        
        const response = await fetch(`${API_BASE}/pdp-analysis`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || '生成PDP失败');
        }
        
        const data = await response.json();
        displayPDPResults(data);
        
    } catch (error) {
        showError('生成部分依赖图失败: ' + error.message);
    } finally {
        showLoading(false);
    }
}

// 显示PDP分析结果
function displayPDPResults(data) {
    const container = document.getElementById('pdp-results');
    
    const html = `
        <div class="pdp-plot-container">
            <h4>部分依赖图</h4>
            <img src="data:image/png;base64,${data.plot_image}" alt="部分依赖图" />
        </div>
        
        <div class="pdp-info">
            <h5>图表说明</h5>
            <ul>
                <li><strong>X轴:</strong> 特征值范围</li>
                <li><strong>Y轴:</strong> 预测概率的变化</li>
                <li><strong>绿色曲线:</strong> 部分依赖关系</li>
                ${currentPDPSample ? '<li><strong>红色点:</strong> 当前样本的特征值位置</li>' : ''}
                <li><strong>解释:</strong> 曲线显示了每个特征对模型预测结果的独立影响</li>
            </ul>
        </div>
    `;
    
    container.innerHTML = html;
}