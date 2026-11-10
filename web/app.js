// Quản lý phiên tìm kiếm và tương tác giao diện người dùng
(function() {
  function getSessionId() {
    let sid = null;
    try {
      sid = localStorage.getItem('session_id');
    } catch (e) {
      console.error('Lỗi khi truy cập localStorage:', e);
    }
    if (!sid) {
      if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
        sid = crypto.randomUUID();
      } else {
        sid = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
          const r = (Math.random() * 16) | 0;
          const v = c === 'x' ? r : (r & 0x3) | 0x8;
          return v.toString(16);
        });
      }
      try {
        localStorage.setItem('session_id', sid);
      } catch (e) {
        console.error('Lỗi khi ghi localStorage:', e);
      }
    }
    return sid;
  }

  const sessionId = getSessionId();
  let currentQueryId = null;

  const qInput = document.getElementById('q');
  const searchBtn = document.getElementById('search-btn');
  const resultsDiv = document.getElementById('results');
  const togglePersonalize = document.getElementById('toggle-personalize');
  const toggleDebug = document.getElementById('toggle-debug');
  const clearHistoryBtn = document.getElementById('clear-history');
  const timingSpan = document.getElementById('timing');
  const clickCountSpan = document.getElementById('click-count');

  async function updateSessionStats() {
    try {
      const resp = await fetch('/session', {
        headers: {
          'X-Session-Id': sessionId
        }
      });
      if (resp.ok) {
        const data = await resp.json();
        clickCountSpan.textContent = data.event_count || 0;
      }
    } catch (err) {
      console.error('Lỗi khi lấy thông tin phiên:', err);
    }
  }

  async function performSearch() {
    const query = qInput.value.trim();
    if (!query) return;

    const personalize = togglePersonalize.checked;
    const debug = toggleDebug.checked;
    const url = `/search?q=${encodeURIComponent(query)}&personalize=${personalize}&debug=${debug}`;

    try {
      const resp = await fetch(url, {
        headers: {
          'X-Session-Id': sessionId
        }
      });
      if (!resp.ok) {
        resultsDiv.innerHTML = `<div class="error">Lỗi khi tìm kiếm (mã ${resp.status})</div>`;
        return;
      }
      const data = await resp.json();
      currentQueryId = data.query_id;
      timingSpan.textContent = `${data.timing_ms && data.timing_ms.total ? data.timing_ms.total : 0} ms`;

      renderResults(data.results || [], debug);
    } catch (err) {
      console.error('Lỗi khi gọi API tìm kiếm:', err);
      resultsDiv.innerHTML = '<div class="error">Không thể kết nối đến máy chủ tìm kiếm.</div>';
    }
  }

  function renderResults(results, isDebug) {
    resultsDiv.innerHTML = '';
    if (results.length === 0) {
      resultsDiv.innerHTML = '<div>Không tìm thấy sản phẩm nào phù hợp.</div>';
      return;
    }

    results.forEach(item => {
      const itemEl = document.createElement('div');
      itemEl.className = 'result';
      itemEl.setAttribute('data-uid', item.product_uid);
      itemEl.setAttribute('data-position', item.position);

      const titleEl = document.createElement('div');
      titleEl.className = 'title';
      titleEl.textContent = item.title;

      if (item.explored) {
        const badge = document.createElement('span');
        badge.className = 'explored-badge';
        badge.textContent = 'Khám phá';
        titleEl.appendChild(badge);
      }
      itemEl.appendChild(titleEl);

      if (item.brand) {
        const brandEl = document.createElement('div');
        brandEl.className = 'brand';
        brandEl.textContent = item.brand;
        itemEl.appendChild(brandEl);
      }

      if (item.snippet) {
        const snippetEl = document.createElement('div');
        snippetEl.className = 'snippet';
        snippetEl.textContent = item.snippet;
        itemEl.appendChild(snippetEl);
      }

      if (isDebug && item.debug) {
        const debugBox = document.createElement('div');
        debugBox.className = 'debug-box';
        debugBox.textContent = JSON.stringify(item.debug);
        itemEl.appendChild(debugBox);
      }

      itemEl.addEventListener('click', async () => {
        itemEl.classList.add('viewed');
        if (currentQueryId) {
          try {
            await fetch('/events', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
                'X-Session-Id': sessionId
              },
              body: JSON.stringify({
                query_id: currentQueryId,
                product_uid: item.product_uid,
                type: 'click',
                position: item.position
              })
            });
            await updateSessionStats();
          } catch (e) {
            console.error('Lỗi khi gửi sự kiện click:', e);
          }
        }
      });

      resultsDiv.appendChild(itemEl);
    });
  }

  searchBtn.addEventListener('click', performSearch);
  qInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      performSearch();
    }
  });

  clearHistoryBtn.addEventListener('click', async () => {
    try {
      await fetch('/session', {
        method: 'DELETE',
        headers: {
          'X-Session-Id': sessionId
        }
      });
      await updateSessionStats();
    } catch (e) {
      console.error('Lỗi khi gọi DELETE /session:', e);
    }
  });

  // Tải trạng thái ban đầu của phiên
  updateSessionStats();
})();
