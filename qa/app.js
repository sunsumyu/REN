document.addEventListener('DOMContentLoaded', () => {
    // Initial bootstrap loads the dataset explorer
    fetchDataset();
});

// ==========================================================
// STATE MANAGEMENT
// ==========================================================
let allRounds = [];
let allPrompts = [];
let currentPromptIndex = 0;
let currentPromptHistory = [];
let activeTab = 'dataset';

// ==========================================================
// TAB SWITCHING NAVIGATION
// ==========================================================
function switchTab(tabName) {
    if (tabName === activeTab) return;
    activeTab = tabName;

    // Toggle active tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById(`tab-${tabName}`).classList.add('active');

    // Toggle active layouts
    document.querySelectorAll('.app-layout').forEach(layout => layout.classList.remove('active'));
    document.getElementById(`layout-${tabName}`).classList.add('active');

    // Load data for the selected tab if needed
    if (tabName === 'prompt' && allPrompts.length === 0) {
        fetchPrompts();
    }
}

// ==========================================================
// TAB 1: DATASET EXPLORER LOGIC
// ==========================================================
async function fetchDataset() {
    try {
        const response = await fetch('medical_qa_dataset.json');
        if (!response.ok) throw new Error('Network response was not ok');
        const data = await response.json();
        
        allRounds = [];
        if (data.history && Array.isArray(data.history)) {
            data.history.forEach(h => allRounds.push(h));
        }
        if (data.Q) {
            allRounds.push(data);
        }

        if (allRounds.length === 0) {
            throw new Error('No valid rounds found in dataset');
        }

        renderDatasetSidebar();
        
        // Hide loader and show content
        document.getElementById('loader').style.opacity = '0';
        setTimeout(() => {
            document.getElementById('loader').style.display = 'none';
            document.getElementById('content-wrapper').style.display = 'block';
            // Render latest round by default
            renderRound(allRounds.length - 1);
        }, 400);

    } catch (error) {
        console.error('Failed to load dataset:', error);
        document.getElementById('loader').innerHTML = `
            <div style="color: #ef4444; text-align: center; max-width: 500px; padding: 20px;">
                <h2>⚠️ 数据集加载失败</h2>
                <p style="margin-top: 12px; color: var(--text-muted);">可能还未生成对话数据，或者浏览器 CORS 限制了本地静态文件读取。</p>
                <p style="margin-top: 16px; font-size: 0.9em; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 8px; border: 1px dashed var(--glass-border);">
                    请通过本站自带的 Web 控制台服务器进行浏览：<br/>
                    <strong style="color: var(--accent-bright);">python qa/web_server.py</strong>
                </p>
            </div>`;
    }
}

function renderDatasetSidebar() {
    const list = document.getElementById('round-list');
    list.innerHTML = '';
    
    allRounds.forEach((round, index) => {
        const div = document.createElement('div');
        div.className = 'round-item';
        div.innerHTML = `
            <div class="round-item-title">第 ${index + 1} 轮对话${index === allRounds.length - 1 ? ' (最新)' : ''}</div>
            <div class="round-item-preview">${round.Q}</div>
        `;
        div.onclick = () => renderRound(index);
        list.appendChild(div);
    });
}

function renderRound(index) {
    const items = document.querySelectorAll('#round-list .round-item');
    items.forEach(i => i.classList.remove('active'));
    if (items[index]) items[index].classList.add('active');

    const round = allRounds[index];

    // Trigger animation re-flow for seamless transitions
    const wrapper = document.getElementById('content-wrapper');
    wrapper.style.animation = 'none';
    wrapper.offsetHeight; /* trigger reflow */
    wrapper.style.animation = null; 

    // Render Q
    document.getElementById('question-text').textContent = round.Q || "No question provided";

    // Render Facets
    const grid = document.getElementById('facets-grid');
    grid.innerHTML = '';
    
    if (round.planners && round.planners.length > 0) {
        round.planners.forEach(p => {
            const card = document.createElement('div');
            card.className = 'facet-card';

            const rawAnswer = p.answer || "";
            let thinkContent = "No reasoning process generated.";
            let actualAnswer = rawAnswer;
            
            const thinkEndIdx = rawAnswer.indexOf('</think>');
            if (rawAnswer.includes('<think') && thinkEndIdx !== -1) {
                const thinkStartIdx = rawAnswer.indexOf('>') + 1;
                thinkContent = rawAnswer.substring(thinkStartIdx, thinkEndIdx).trim();
                actualAnswer = rawAnswer.substring(thinkEndIdx + 8).trim();
            }

            // Detect potential hallucinations (business topics leaking into medical QAs)
            const isSuspicious = (actualAnswer.includes('数据隐私') || actualAnswer.includes('供应链') || actualAnswer.includes('项目管理') || actualAnswer.includes('合规审计')) && !round.Q.includes('企业');
            const badge = isSuspicious ? `<div class="error-badge">⚠️ 检测到非医学潜在幻觉 (示例词污染)</div>` : '';

            card.innerHTML = `
                <div class="facet-header">
                    <div class="facet-title">${p.planner}</div>
                </div>
                <div class="think-block">
                    <div class="think-header" onclick="this.nextElementSibling.classList.toggle('open')">
                        <span><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align: middle; margin-right: 4px;"><circle cx="12" cy="12" r="10"></circle><polyline points="12 16 12 12 12 8"></polyline><line x1="12" y1="16" x2="12.01" y2="16"></line></svg> 显式证据推理链 (Evidence Chain)</span>
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"></polyline></svg>
                    </div>
                    <div class="think-content">${escapeHtml(thinkContent)}</div>
                </div>
                <div class="facet-body markdown-body">
                    ${marked.parse(actualAnswer)}
                    ${badge}
                </div>
            `;
            grid.appendChild(card);
        });
    } else {
        grid.innerHTML = '<div style="color: var(--text-muted); grid-column: 1/-1; text-align: center; padding: 40px;">暂无该轮次的角度拆解数据。</div>';
    }

    // Render Summary
    const summaryPanel = document.getElementById('summary-content');
    if (round.summary) {
        summaryPanel.innerHTML = marked.parse(round.summary);
    } else {
        summaryPanel.innerHTML = '<em style="color: var(--text-muted)">暂无综合总结。</em>';
    }
}

// ==========================================================
// TAB 2: PROMPT HUB CORE LOGIC
// ==========================================================
async function fetchPrompts() {
    try {
        const response = await fetch('/api/prompts');
        if (!response.ok) throw new Error('API fetch failed');
        const resJson = await response.json();
        
        if (resJson.success) {
            allPrompts = resJson.data;
            renderPromptSidebar();
            if (allPrompts.length > 0) {
                selectPrompt(0);
            }
        } else {
            throw new Error(resJson.msg);
        }
    } catch (error) {
        console.error('Failed to fetch prompts:', error);
        showToast('获取提示词模版失败，请确保 Web 服务运行正常。', 'error');
    }
}

function renderPromptSidebar() {
    const list = document.getElementById('prompt-list');
    list.innerHTML = '';
    
    allPrompts.forEach((p, index) => {
        const div = document.createElement('div');
        div.className = 'round-item';
        div.innerHTML = `
            <div class="round-item-title" style="font-size:0.85rem; font-family: monospace;">${p.name}</div>
            <div class="round-item-preview" style="color: var(--accent-bright);">当前版本: V${p.active_version} | 共 ${p.total_versions} 个版本</div>
        `;
        div.onclick = () => selectPrompt(index);
        list.appendChild(div);
    });
}

async function selectPrompt(index) {
    currentPromptIndex = index;
    const items = document.querySelectorAll('#prompt-list .round-item');
    items.forEach(i => i.classList.remove('active'));
    if (items[index]) items[index].classList.add('active');

    const promptObj = allPrompts[index];
    document.getElementById('current-prompt-title').textContent = promptObj.name;
    document.getElementById('active-version-badge').textContent = `Version ${promptObj.active_version}`;

    // Clear form inputs
    document.getElementById('update-description').value = '';

    // Fetch version timeline history
    try {
        const response = await fetch(`/api/prompts/history?name=${promptObj.name}`);
        if (!response.ok) throw new Error('History fetch failed');
        const resJson = await response.json();

        if (resJson.success) {
            currentPromptHistory = resJson.data;
            renderTimeline();
            
            // Find currently active content and bind to editor
            const activePrompt = currentPromptHistory.find(v => v.is_active);
            if (activePrompt) {
                document.getElementById('prompt-editor').value = activePrompt.content;
            }
        }
    } catch (error) {
        console.error('Failed to load history:', error);
        showToast('读取版本时光机历史失败。', 'error');
    }
}

function renderTimeline() {
    const timeline = document.getElementById('version-timeline');
    timeline.innerHTML = '';

    currentPromptHistory.forEach((v) => {
        const div = document.createElement('div');
        div.className = `timeline-item ${v.is_active ? 'active' : ''}`;
        
        const activeBadge = v.is_active ? `<span class="timeline-badge-active">当前激活</span>` : '';
        const rollbackBtn = !v.is_active ? `<button class="rollback-btn" onclick="triggerRollback(${v.version})">↩️ 激活此版本</button>` : '';

        div.innerHTML = `
            <div class="timeline-dot"></div>
            <div class="timeline-header">
                <span>Version ${v.version}</span>
                ${activeBadge}
            </div>
            <div class="timeline-notes">${escapeHtml(v.description || '无更新说明')}</div>
            <div class="timeline-time">${v.created_at}</div>
            ${rollbackBtn}
        `;
        timeline.appendChild(div);
    });
}

// Save dynamic prompt update
async function savePromptUpdate() {
    const promptObj = allPrompts[currentPromptIndex];
    const editorContent = document.getElementById('prompt-editor').value.trim();
    const updateDesc = document.getElementById('update-description').value.trim();

    if (!editorContent) {
        showToast('提示词模板内容不能为空！', 'error');
        return;
    }

    if (!updateDesc) {
        showToast('为了维护规范，请填写本次版本的更新说明！', 'error');
        return;
    }

    try {
        const response = await fetch('/api/prompts/update', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name: promptObj.name,
                content: editorContent,
                description: updateDesc
            })
        });
        
        const resJson = await response.json();
        if (resJson.success) {
            showToast(`成功创建并激活 ${promptObj.name} 新版本！`, 'success');
            // Refresh list and reload details
            await fetchPrompts();
            selectPrompt(currentPromptIndex);
        } else {
            showToast(`保存失败: ${resJson.msg}`, 'error');
        }
    } catch (error) {
        console.error('Failed to submit prompt update:', error);
        showToast('与后端服务器通信异常，保存失败。', 'error');
    }
}

// Rollback dynamic prompt version
async function triggerRollback(versionNum) {
    const promptObj = allPrompts[currentPromptIndex];
    
    try {
        const response = await fetch('/api/prompts/rollback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name: promptObj.name,
                version: versionNum
            })
        });
        
        const resJson = await response.json();
        if (resJson.success) {
            showToast(`成功将 ${promptObj.name} 回滚激活至 Version ${versionNum}！`, 'success');
            await fetchPrompts();
            selectPrompt(currentPromptIndex);
        } else {
            showToast(`回滚失败: ${resJson.msg}`, 'error');
        }
    } catch (error) {
        console.error('Failed to execute rollback:', error);
        showToast('与后端服务器通信异常，回滚失败。', 'error');
    }
}

// ==========================================================
// TOAST ALERTS & UTILS
// ==========================================================
function showToast(message, type = 'success') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    
    const svgIcon = type === 'success' 
        ? `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>`
        : `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>`;

    toast.innerHTML = `
        ${svgIcon}
        <span>${message}</span>
    `;
    
    container.appendChild(toast);
    
    // Smooth slide-in
    setTimeout(() => {
        toast.classList.add('show');
    }, 50);

    // Auto dismiss after 3s
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => {
            toast.remove();
        }, 400);
    }, 3000);
}

function escapeHtml(unsafe) {
    if (!unsafe) return '';
    return unsafe
         .replace(/&/g, "&amp;")
         .replace(/</g, "&lt;")
         .replace(/>/g, "&gt;")
         .replace(/"/g, "&quot;")
         .replace(/'/g, "&#039;");
}
