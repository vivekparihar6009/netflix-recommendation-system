// Global App State
let allMovies = []; // Loaded from API for autocomplete
let sandboxRatings = {}; // Holds { movie_id: rating }
let metricsChartInstance = null;

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    loadDashboardStats();
    loadMovieList();
    initRecsSection();
    initSimilaritySection();
    initSandboxSection();
});

// 1. Tab Navigation
function initNavigation() {
    const navButtons = document.querySelectorAll('.nav-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    const pageTitle = document.getElementById('page-title');
    const pageSubtitle = document.getElementById('page-subtitle');
    
    const titleMap = {
        'dashboard-section': { title: 'Dashboard Overview', subtitle: 'Evaluation metrics, model comparisons, and dataset details.' },
        'eda-section': { title: 'Exploratory Data Analysis (EDA)', subtitle: 'Analysis of dataset distributions, temporal trends, and model learning curves.' },
        'recommend-section': { title: 'Personalized Recommendations', subtitle: 'Predict ratings and generate top-10 recommended titles for any user.' },
        'similarity-section': { title: 'Movie Similarity Explorer', subtitle: 'Find similar films based on user viewing preferences and collaborative filtering similarity vectors.' },
        'sandbox-section': { title: 'New User Personalization Sandbox', subtitle: 'Create a temporary profile, rate movies, and see real-time recommender outputs.' }
    };
    
    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const target = btn.getAttribute('data-target');
            
            // Toggle button active class
            navButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Toggle tab display
            tabContents.forEach(tab => {
                tab.classList.remove('active');
                if (tab.id === target) {
                    tab.classList.add('active');
                }
            });
            
            // Update Page Headers
            if (titleMap[target]) {
                pageTitle.textContent = titleMap[target].title;
                pageSubtitle.textContent = titleMap[target].subtitle;
            }
        });
    });
}

// 2. Fetch Dashboard Stats & Build Evaluation Chart
function loadDashboardStats() {
    fetch('/api/stats')
        .then(res => res.json())
        .then(data => {
            if (data.status === 'success') {
                // Populate Stat Cards
                document.getElementById('stat-users').textContent = Number(data.stats.users).toLocaleString();
                document.getElementById('stat-movies').textContent = Number(data.stats.movies).toLocaleString();
                document.getElementById('stat-ratings').textContent = Number(data.stats.ratings).toLocaleString();
                document.getElementById('stat-sparsity').textContent = data.stats.sparsity;
                
                // Populate Table
                buildMetricsTable(data.metrics);
                
                // Render Status Tags
                buildStatusTags(data.models_status);
                
                // Draw Chart
                renderMetricsChart(data.metrics);
            }
        })
        .catch(err => console.error('Error fetching dashboard stats:', err));
}

function buildMetricsTable(metrics) {
    const tableBody = document.querySelector('#metrics-table tbody');
    tableBody.innerHTML = '';
    
    if (!metrics || Object.keys(metrics).length === 0) {
        tableBody.innerHTML = `<tr><td colspan="7" class="loading-cell">No metrics found. Run 'verify_pipeline.py' or train pipeline first.</td></tr>`;
        return;
    }
    
    // Ordered models
    const models = ['SVD', 'User-CF', 'Item-CF', 'Hybrid'];
    
    models.forEach(model => {
        const rowData = metrics[model];
        if (!rowData) return;
        
        const row = document.createElement('tr');
        
        // Find best (bold) values in row logic
        const isSvd = model === 'SVD';
        const isHybrid = model === 'Hybrid';
        
        row.innerHTML = `
            <td><strong>${model}</strong></td>
            <td>${isSvd ? `<strong>${rowData.RMSE.toFixed(4)}</strong>` : rowData.RMSE.toFixed(4)}</td>
            <td>${isSvd ? `<strong>${rowData.MAE.toFixed(4)}</strong>` : rowData.MAE.toFixed(4)}</td>
            <td>${isHybrid ? `<strong>${rowData['MAP@10'].toFixed(4)}</strong>` : rowData['MAP@10'].toFixed(4)}</td>
            <td>${isHybrid ? `<strong>${rowData['Precision@10'].toFixed(4)}</strong>` : rowData['Precision@10'].toFixed(4)}</td>
            <td>${isHybrid ? `<strong>${rowData['Recall@10'].toFixed(4)}</strong>` : rowData['Recall@10'].toFixed(4)}</td>
            <td>${isHybrid ? `<strong>${rowData['NDCG@10'].toFixed(4)}</strong>` : rowData['NDCG@10'].toFixed(4)}</td>
        `;
        tableBody.appendChild(row);
    });
}

function buildStatusTags(status) {
    const container = document.getElementById('model-status-tags');
    container.innerHTML = '';
    
    const models = [
        { name: 'SVD Model', active: status.svd },
        { name: 'User-CF Model', active: status.user_cf },
        { name: 'Item-CF Model', active: status.item_cf },
        { name: 'Hybrid Recommender', active: status.hybrid }
    ];
    
    models.forEach(m => {
        const tag = document.createElement('span');
        tag.className = `model-tag ${m.active ? 'active' : 'inactive'}`;
        tag.innerHTML = m.active 
            ? `<i class="fa-solid fa-circle-check"></i> ${m.name}: Loaded`
            : `<i class="fa-solid fa-circle-xmark"></i> ${m.name}: Missing`;
        container.appendChild(tag);
    });
}

function renderMetricsChart(metrics) {
    const ctx = document.getElementById('metricsChart').getContext('2d');
    
    if (!metrics || Object.keys(metrics).length === 0) return;
    
    const models = ['SVD', 'User-CF', 'Item-CF', 'Hybrid'];
    const metricsKeys = ['RMSE', 'MAP@10', 'Precision@10', 'NDCG@10'];
    
    const colors = {
        'SVD': 'rgba(229, 9, 20, 0.85)',       // Netflix Red
        'User-CF': 'rgba(54, 162, 235, 0.85)',   // Blue
        'Item-CF': 'rgba(255, 206, 86, 0.85)',   // Yellow
        'Hybrid': 'rgba(29, 185, 84, 0.85)'     // Green
    };
    
    const datasets = models.map(model => {
        return {
            label: model,
            data: metricsKeys.map(key => metrics[model] ? metrics[model][key] : 0),
            backgroundColor: colors[model],
            borderColor: colors[model].replace('0.85', '1.0'),
            borderWidth: 1
        };
    });
    
    if (metricsChartInstance) {
        metricsChartInstance.destroy();
    }
    
    metricsChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['RMSE (Lower is Better)', 'MAP@10', 'Precision@10', 'NDCG@10'],
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#f0f0f5', font: { family: 'Outfit', size: 12 } }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#8a8a9a', font: { family: 'Outfit' } }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#8a8a9a', font: { family: 'Outfit' } },
                    min: 0,
                    max: 1.0
                }
            }
        }
    });
}

// 3. Load Autocomplete Movie List
function loadMovieList() {
    fetch('/api/movies')
        .then(res => res.json())
        .then(data => {
            if (data.status === 'success') {
                allMovies = data.movies;
            }
        })
        .catch(err => console.error('Error loading movie list autocomplete:', err));
}

// 4. Personalized Recommendations Tab logic
function initRecsSection() {
    const btnGetRecs = document.getElementById('btn-get-recs');
    const userInput = document.getElementById('recs-user-id');
    const modelSelect = document.getElementById('recs-model');
    const resultsContainer = document.getElementById('recs-results-container');
    const emptyState = document.getElementById('recs-empty-state');
    const movieGrid = document.getElementById('recs-movie-grid');
    const profileTitle = document.getElementById('recs-profile-title');
    const profileDesc = document.getElementById('recs-profile-desc');
    const activeModelBadge = document.getElementById('recs-active-model');
    
    btnGetRecs.addEventListener('click', () => {
        const userIdVal = userInput.value.trim ? userInput.value.trim() : userInput.value;
        if (!userIdVal) {
            alert('Please enter a User ID.');
            return;
        }
        
        const userId = parseInt(userIdVal);
        const modelType = modelSelect.value;
        
        movieGrid.innerHTML = '<div class="loading-cell">Calculating recommendations...</div>';
        emptyState.style.display = 'none';
        resultsContainer.style.display = 'block';
        
        fetch(`/api/recommend?user_id=${userId}&model_type=${modelType}`)
            .then(res => res.json())
            .then(data => {
                if (data.status === 'success') {
                    // Update header
                    profileTitle.textContent = `User #${data.user_id}`;
                    if (data.is_known) {
                        profileDesc.textContent = `Status: Active User (${data.ratings_count} ratings in training set)`;
                    } else {
                        profileDesc.textContent = `Status: This user has 0 movie ratings in the dataset. Since we don't know their preferences yet, we are recommending the most popular and highly-rated movies on the platform.`;
                    }
                    activeModelBadge.textContent = `${data.model_type.toUpperCase()} MODEL`;
                    
                    // Populate Movie Grid
                    renderMovieCards(data.recommendations, movieGrid, userId);
                } else {
                    movieGrid.innerHTML = `<div class="loading-cell text-red">Error: ${data.message}</div>`;
                }
            })
            .catch(err => {
                console.error(err);
                movieGrid.innerHTML = '<div class="loading-cell text-red">Server communication error.</div>';
            });
    });
}

function selectUser(uid) {
    document.getElementById('recs-user-id').value = uid;
    document.getElementById('btn-get-recs').click();
}

function renderMovieCards(recs, container, userId) {
    container.innerHTML = '';
    
    if (!recs || recs.length === 0) {
        container.innerHTML = '<div class="loading-cell">No recommendations generated.</div>';
        return;
    }
    
    recs.forEach((rec, idx) => {
        const card = document.createElement('div');
        card.className = 'movie-card';
        
        const yearStr = rec.year > 0 ? rec.year : 'N/A';
        
        card.innerHTML = `
            <div class="card-rank">#${idx + 1}</div>
            <div class="movie-card-header">
                <div class="card-title">${rec.title}</div>
                <div class="card-year">${yearStr}</div>
            </div>
            <div class="movie-card-details">
                <div class="rating-badge">
                    <i class="fa-solid fa-star"></i>
                    <span>${rec.predicted_rating.toFixed(2)}</span>
                </div>
                <button class="card-btn" onclick="explainRec(${userId}, ${rec.movie_id})">
                    <i class="fa-solid fa-circle-info"></i> Explain
                </button>
            </div>
        `;
        container.appendChild(card);
    });
}

// Explanation modal handler
window.explainRec = function(userId, movieId) {
    const modal = document.getElementById('explanation-modal');
    const modalTitle = document.getElementById('modal-movie-title');
    const modalContent = document.getElementById('modal-explanation-content');
    
    modalTitle.textContent = 'Loading...';
    modalContent.textContent = 'Fetching explanation logic from similarity mapper...';
    modal.classList.add('active');
    
    fetch(`/api/explain?user_id=${userId}&movie_id=${movieId}`)
        .then(res => res.json())
        .then(data => {
            if (data.status === 'success') {
                modalTitle.textContent = data.movie_title;
                modalContent.textContent = data.explanation;
            } else {
                modalContent.textContent = `Error: ${data.message}`;
            }
        })
        .catch(err => {
            modalContent.textContent = 'Failed to fetch explanation.';
            console.error(err);
        });
};

window.closeModal = function() {
    document.getElementById('explanation-modal').classList.remove('active');
};

// 5. Movie Similarity Section Tab logic
function initSimilaritySection() {
    const searchInput = document.getElementById('similarity-search');
    const btnFindSimilar = document.getElementById('btn-find-similar');
    const resultsContainer = document.getElementById('similar-results-container');
    const emptyState = document.getElementById('similar-empty-state');
    const movieGrid = document.getElementById('similar-movie-grid');
    const targetTitleHeader = document.getElementById('similar-target-title');
    
    // Autocomplete list elements
    const autocompleteList = document.getElementById('autocomplete-list');
    
    setupAutocomplete(searchInput, autocompleteList, (selectedMovie) => {
        searchInput.value = selectedMovie.title;
        btnFindSimilar.disabled = false;
        btnFindSimilar.setAttribute('data-movie-title', selectedMovie.title);
    });
    
    btnFindSimilar.addEventListener('click', () => {
        const movieTitle = btnFindSimilar.getAttribute('data-movie-title');
        if (!movieTitle) return;
        
        movieGrid.innerHTML = '<div class="loading-cell">Loading similar movies...</div>';
        emptyState.style.display = 'none';
        resultsContainer.style.display = 'block';
        
        fetch(`/api/similar?title=${encodeURIComponent(movieTitle)}`)
            .then(res => res.json())
            .then(data => {
                if (data.status === 'success') {
                    targetTitleHeader.textContent = data.target_title;
                    
                    // Render list
                    movieGrid.innerHTML = '';
                    if (data.similar_movies.length === 0) {
                        movieGrid.innerHTML = '<div class="loading-cell">No highly similar movies found.</div>';
                        return;
                    }
                    
                    data.similar_movies.forEach((rec, idx) => {
                        const card = document.createElement('div');
                        card.className = 'movie-card';
                        
                        card.innerHTML = `
                            <div class="card-rank">#${idx + 1}</div>
                            <div class="movie-card-header">
                                <div class="card-title">${rec.title}</div>
                                <div class="card-year">${rec.year > 0 ? rec.year : 'N/A'}</div>
                            </div>
                            <div class="movie-card-details">
                                <div class="sim-percentage">
                                    <i class="fa-solid fa-link"></i> ${Math.round(rec.similarity_score * 100)}% Match
                                </div>
                            </div>
                        `;
                        movieGrid.appendChild(card);
                    });
                } else {
                    movieGrid.innerHTML = `<div class="loading-cell text-red">Error: ${data.message}</div>`;
                }
            })
            .catch(err => {
                console.error(err);
                movieGrid.innerHTML = '<div class="loading-cell text-red">Server communication error.</div>';
            });
    });
}

// 6. New User Sandbox Tab logic
function initSandboxSection() {
    const searchInput = document.getElementById('sandbox-search');
    const autocompleteList = document.getElementById('sandbox-autocomplete-list');
    const ratedListContainer = document.getElementById('rated-movies-list');
    const btnSandboxRecs = document.getElementById('btn-sandbox-recs');
    
    const resultsContainer = document.getElementById('sandbox-recs-container');
    const emptyState = document.getElementById('sandbox-empty-state');
    const recsListElement = document.getElementById('sandbox-recs-list');
    const explanationElement = document.getElementById('sandbox-explanation');
    
    setupAutocomplete(searchInput, autocompleteList, (selectedMovie) => {
        searchInput.value = '';
        promptRatingWidget(selectedMovie, (ratingVal) => {
            sandboxRatings[selectedMovie.id] = ratingVal;
            updateRatedListUI(ratedListContainer);
            btnSandboxRecs.disabled = false;
        });
    });
    
    btnSandboxRecs.addEventListener('click', () => {
        if (Object.keys(sandboxRatings).length === 0) return;
        
        recsListElement.innerHTML = '<div class="loading-cell">Calculating real-time updates...</div>';
        emptyState.style.display = 'none';
        resultsContainer.style.display = 'block';
        
        fetch('/api/rate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ratings: sandboxRatings })
        })
        .then(res => res.json())
        .then(data => {
            if (data.status === 'success') {
                explanationElement.textContent = data.top_explanation;
                
                // Populate List
                recsListElement.innerHTML = '';
                if (data.recommendations.length === 0) {
                    recsListElement.innerHTML = '<div class="loading-cell">No recommendations found. Try rating more movies.</div>';
                    return;
                }
                
                data.recommendations.forEach(rec => {
                    const item = document.createElement('div');
                    item.className = 'sandbox-rec-item';
                    item.innerHTML = `
                        <div>
                            <span class="sandbox-rec-title">${rec.title}</span>
                            <span class="sandbox-rec-year">(${rec.year})</span>
                        </div>
                        <div class="sandbox-rec-score">
                            <i class="fa-solid fa-star"></i> ${rec.predicted_rating.toFixed(2)}
                        </div>
                    `;
                    recsListElement.appendChild(item);
                });
            } else {
                recsListElement.innerHTML = `<div class="loading-cell text-red">Error: ${data.message}</div>`;
            }
        })
        .catch(err => {
            console.error(err);
            recsListElement.innerHTML = '<div class="loading-cell text-red">Server communication error.</div>';
        });
    });
}

function promptRatingWidget(movie, onSelectRating) {
    // Dynamically insert a rating selection popup box inside the ratedMoviesList container
    const container = document.getElementById('rated-movies-list');
    
    // Remove placeholder
    const placeholder = container.querySelector('.no-ratings-placeholder');
    if (placeholder) placeholder.style.display = 'none';
    
    const promptItem = document.createElement('div');
    promptItem.className = 'rated-item prompt-item';
    promptItem.innerHTML = `
        <div class="rated-movie-info">
            <h4>${movie.title}</h4>
            <span>Select stars to add to profile:</span>
        </div>
        <div class="rated-right">
            <div class="star-rating-widget" id="temp-star-widget">
                <i class="fa-solid fa-star" data-val="1"></i>
                <i class="fa-solid fa-star" data-val="2"></i>
                <i class="fa-solid fa-star" data-val="3"></i>
                <i class="fa-solid fa-star" data-val="4"></i>
                <i class="fa-solid fa-star" data-val="5"></i>
            </div>
        </div>
    `;
    container.insertBefore(promptItem, container.firstChild);
    
    // Wire up star click events
    const stars = promptItem.querySelectorAll('.star-rating-widget i');
    stars.forEach(star => {
        star.addEventListener('mouseover', () => {
            const hoverVal = parseInt(star.getAttribute('data-val'));
            stars.forEach(s => {
                const sVal = parseInt(s.getAttribute('data-val'));
                s.className = sVal <= hoverVal ? 'fa-solid fa-star selected' : 'fa-solid fa-star';
            });
        });
        
        star.addEventListener('click', () => {
            const finalVal = parseFloat(star.getAttribute('data-val'));
            promptItem.remove();
            onSelectRating(finalVal);
        });
    });
    
    // Reset stars layout when mouse leaves widget
    const widget = promptItem.querySelector('#temp-star-widget');
    widget.addEventListener('mouseleave', () => {
        stars.forEach(s => s.className = 'fa-solid fa-star');
    });
}

function updateRatedListUI(container) {
    // Clear list but keep search input
    container.innerHTML = '';
    const keys = Object.keys(sandboxRatings);
    
    if (keys.length === 0) {
        container.innerHTML = `
            <div class="no-ratings-placeholder">
                <i class="fa-solid fa-folder-open"></i>
                <p>No ratings added yet. Search a movie above and select a star rating to populate your profile.</p>
            </div>
        `;
        document.getElementById('btn-sandbox-recs').disabled = true;
        return;
    }
    
    keys.forEach(midStr => {
        const mid = parseInt(midStr);
        const movieObj = allMovies.find(m => m.id === mid);
        const rating = sandboxRatings[mid];
        
        if (!movieObj) return;
        
        const item = document.createElement('div');
        item.className = 'rated-item';
        
        let starsStr = '';
        for (let i = 1; i <= 5; i++) {
            starsStr += i <= rating 
                ? '<i class="fa-solid fa-star"></i>' 
                : '<i class="fa-regular fa-star"></i>';
        }
        
        item.innerHTML = `
            <div class="rated-movie-info">
                <h4>${movieObj.title}</h4>
                <span>Movie ID: ${mid} (${movieObj.year})</span>
            </div>
            <div class="rated-right">
                <div class="rated-stars">${starsStr}</div>
                <button class="btn-remove-rated" onclick="removeSandboxRating(${mid})">
                    <i class="fa-solid fa-trash"></i>
                </button>
            </div>
        `;
        container.appendChild(item);
    });
}

window.removeSandboxRating = function(mid) {
    delete sandboxRatings[mid];
    updateRatedListUI(document.getElementById('rated-movies-list'));
};

// Autocomplete Component Helper
function setupAutocomplete(input, listContainer, onSelect) {
    let currentFocus;
    
    input.addEventListener('input', function(e) {
        const val = this.value;
        closeAllLists();
        if (!val) return false;
        currentFocus = -1;
        
        // Filter movies matching input
        const matches = allMovies.filter(m => m.title.toLowerCase().includes(val.toLowerCase())).slice(0, 10);
        
        matches.forEach(movie => {
            const item = document.createElement('div');
            // Highlight matching text
            const idx = movie.title.toLowerCase().indexOf(val.toLowerCase());
            const matchLen = val.length;
            const before = movie.title.substring(0, idx);
            const midText = movie.title.substring(idx, idx + matchLen);
            const after = movie.title.substring(idx + matchLen);
            
            item.innerHTML = `${before}<strong>${midText}</strong>${after} <span style="font-size:0.75rem; color:#8a8a9a;">(${movie.year})</span>`;
            
            item.addEventListener('click', function(e) {
                onSelect(movie);
                closeAllLists();
            });
            
            listContainer.appendChild(item);
        });
    });
    
    input.addEventListener('keydown', function(e) {
        let x = listContainer;
        if (x) x = x.getElementsByTagName('div');
        if (e.keyCode === 40) { // Arrow Down
            currentFocus++;
            addActive(x);
        } else if (e.keyCode === 38) { // Arrow Up
            currentFocus--;
            addActive(x);
        } else if (e.keyCode === 13) { // Enter
            e.preventDefault();
            if (currentFocus > -1) {
                if (x) x[currentFocus].click();
            }
        }
    });
    
    function addActive(x) {
        if (!x) return false;
        removeActive(x);
        if (currentFocus >= x.length) currentFocus = 0;
        if (currentFocus < 0) currentFocus = (x.length - 1);
        x[currentFocus].classList.add('autocomplete-active');
    }
    
    function removeActive(x) {
        for (let i = 0; i < x.length; i++) {
            x[i].classList.remove('autocomplete-active');
        }
    }
    
    function closeAllLists(elmnt) {
        listContainer.innerHTML = '';
    }
    
    document.addEventListener('click', function (e) {
        if (e.target !== input && e.target !== listContainer) {
            closeAllLists();
        }
    });
}

// 7. EDA Image Modal Logic
window.openEdaModal = function(src, caption) {
    const modal = document.getElementById('image-modal');
    const modalImg = document.getElementById('modal-img-preview');
    const modalCaption = document.getElementById('modal-img-caption');
    
    modalImg.src = src;
    modalCaption.textContent = caption;
    modal.classList.add('active');
};

window.closeImageModal = function() {
    document.getElementById('image-modal').classList.remove('active');
};

// Wire up backdrop click for image-modal
document.addEventListener('DOMContentLoaded', () => {
    const imgModal = document.getElementById('image-modal');
    if (imgModal) {
        imgModal.addEventListener('click', (e) => {
            if (e.target === imgModal) {
                closeImageModal();
            }
        });
    }
});
