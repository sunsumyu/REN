document.addEventListener('DOMContentLoaded', () => {
    fetchDataset();
});

let allRounds = [];

async function fetchDataset() {
    try {
        const response = await fetch('medical_qa_dataset.json');
        if (!response.ok) throw new Error('Network response was not ok');
        const data = await response.json();
        
        // Flatten rounds: history array + final current state
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

        renderSidebar();
        
        // Hide loader and show content
        document.getElementById('loader').style.opacity = '0';
        setTimeout(() => {
            document.getElementById('loader').style.display = 'none';
            document.getElementById('content-wrapper').style.display = 'block';
            // Render the last round by default
            renderRound(allRounds.length - 1);
        }, 500);

    } catch (error) {
        console.error('Failed to load dataset:', error);
        document.getElementById('loader').innerHTML = `<div style="color: #ef4444; text-align: center;"><h2>Error loading dataset</h2><p>Please ensure you are running a local HTTP server.</p><p style="margin-top: 10px; font-size: 0.9em; opacity: 0.8">python -m http.server 8000</p></div>`;
    }
}

function renderSidebar() {
    const list = document.getElementById('round-list');
    list.innerHTML = '';
    
    allRounds.forEach((round, index) => {
        const div = document.createElement('div');
        div.className = 'round-item';
        div.innerHTML = `
            <div class="round-item-title">Round ${index + 1}${index === allRounds.length - 1 ? ' (Latest)' : ''}</div>
            <div class="round-item-preview">${round.Q}</div>
        `;
        div.onclick = () => renderRound(index);
        list.appendChild(div);
    });
}

function renderRound(index) {
    // Update active state
    const items = document.querySelectorAll('.round-item');
    items.forEach(i => i.classList.remove('active'));
    if(items[index]) items[index].classList.add('active');

    const round = allRounds[index];

    // Trigger animation re-flow
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
            // Split <think> from the rest
            let thinkContent = "No think process generated.";
            let actualAnswer = rawAnswer;
            
            const thinkEndIdx = rawAnswer.indexOf('</think>');
            if (rawAnswer.startsWith('<think') && thinkEndIdx !== -1) {
                const thinkStartIdx = rawAnswer.indexOf('>') + 1;
                thinkContent = rawAnswer.substring(thinkStartIdx, thinkEndIdx).trim();
                actualAnswer = rawAnswer.substring(thinkEndIdx + 8).trim();
            }

            // Check for potential hallucination (if answer contains typical IT/Business keywords but Q is medical)
            const isSuspicious = (actualAnswer.includes('数据隐私') || actualAnswer.includes('供应链') || actualAnswer.includes('项目管理')) && !round.Q.includes('企业');
            const badge = isSuspicious ? `<div class="error-badge">⚠️ Potential Hallucination Detected</div>` : '';

            card.innerHTML = `
                <div class="facet-header">
                    <div class="facet-title">${p.planner}</div>
                </div>
                <div class="think-block">
                    <div class="think-header" onclick="this.nextElementSibling.classList.toggle('open')">
                        <span><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align: middle; margin-right: 4px;"><circle cx="12" cy="12" r="10"></circle><polyline points="12 16 12 12 12 8"></polyline><line x1="12" y1="16" x2="12.01" y2="16"></line></svg> View Reasoning & Evidence</span>
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
        grid.innerHTML = '<div style="color: var(--text-muted);">No facets generated for this round.</div>';
    }

    // Render Summary
    const summaryPanel = document.getElementById('summary-content');
    if (round.summary) {
        summaryPanel.innerHTML = marked.parse(round.summary);
    } else {
        summaryPanel.innerHTML = '<em style="color: var(--text-muted)">No summary available.</em>';
    }
}

function escapeHtml(unsafe) {
    return unsafe
         .replace(/&/g, "&amp;")
         .replace(/</g, "&lt;")
         .replace(/>/g, "&gt;")
         .replace(/"/g, "&quot;")
         .replace(/'/g, "&#039;");
}
