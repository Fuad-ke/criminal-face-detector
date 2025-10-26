// Criminal Face Detection System - JavaScript - FIXED VERSION

let selectedFile = null;

// Initialize on load
document.addEventListener('DOMContentLoaded', function() {
    console.log('Initializing application...');
    initNavigation();
    initImageUpload();
    initForms();
    
    // Load criminals on page load
    console.log('Loading criminals on page load...');
    loadCriminals();
});

// Navigation
function initNavigation() {
    const navButtons = document.querySelectorAll('.nav-btn');
    const sections = document.querySelectorAll('.section');
    
    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetSection = btn.dataset.section;
            
            // Update active button
            navButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Show target section
            sections.forEach(section => {
                if (section.id === `${targetSection}-section`) {
                    section.classList.add('active');
                } else {
                    section.classList.remove('active');
                }
            });
            
            // Load data for specific sections
            if (targetSection === 'criminals') {
                loadCriminals();
            } else if (targetSection === 'statistics') {
                loadStatistics();
            }
        });
    });
}

// Switch between sections
function switchSection(sectionName) {
    // Hide all sections
    document.querySelectorAll('.section').forEach(section => {
        section.classList.remove('active');
    });
    
    // Show target section
    const targetSection = document.getElementById(sectionName + '-section');
    if (targetSection) {
        targetSection.classList.add('active');
    }
    
    // Update nav buttons
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    const targetBtn = document.querySelector(`[data-section="${sectionName}"]`);
    if (targetBtn) {
        targetBtn.classList.add('active');
    }
    
    // Load data for specific sections
    if (sectionName === 'criminals') {
        loadCriminals();
    } else if (sectionName === 'statistics') {
        loadStatistics();
    }
}

// Image Upload
function initImageUpload() {
    const uploadArea = document.getElementById('uploadArea');
    const imageUpload = document.getElementById('imageUpload');
    const detectBtn = document.getElementById('detectBtn');
    
    // Click to upload
    uploadArea.addEventListener('click', () => {
        imageUpload.click();
    });
    
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--primary)';
        uploadArea.style.background = 'rgba(102, 126, 234, 0.1)';
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.borderColor = 'var(--border)';
        uploadArea.style.background = 'transparent';
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--border)';
        uploadArea.style.background = 'transparent';
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleImageSelect(files[0]);
        }
    });
    
    // File input change
    imageUpload.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleImageSelect(e.target.files[0]);
        }
    });
    
    // Detect button
    detectBtn.addEventListener('click', detectCriminals);
}

function handleImageSelect(file) {
    if (!file.type.startsWith('image/')) {
        showToast('Please select an image file', 'error');
        return;
    }
    
    selectedFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        const preview = document.getElementById('imagePreview');
        preview.innerHTML = `<img src="${e.target.result}" alt="Selected image">`;
    };
    reader.readAsDataURL(file);
    
    // Enable detect button
    document.getElementById('detectBtn').disabled = false;
    
    // Update upload area
    const uploadArea = document.getElementById('uploadArea');
    uploadArea.innerHTML = `
        <i class="fas fa-check-circle" style="color: var(--success);"></i>
        <h3>Image selected!</h3>
        <p>Click "Analyze Image" to detect criminals</p>
    `;
}

async function detectCriminals() {
    if (!selectedFile) {
        showToast('Please select an image first', 'error');
        return;
    }
    
    const formData = new FormData();
    formData.append('image', selectedFile);
    
    showLoading(true);
    
    try {
        const response = await fetch('/api/detect/image', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayDetectionResults(data);
            
            if (data.criminals_found > 0) {
                showToast(`⚠️ ${data.criminals_found} CRIMINAL(S) DETECTED!`, 'error');
            } else {
                showToast('No criminals detected in image', 'success');
            }
        } else {
            showToast('Detection failed: ' + data.error, 'error');
        }
    } catch (error) {
        showToast('Error: ' + error.message, 'error');
    } finally {
        showLoading(false);
    }
}

function displayDetectionResults(data) {
    // Display result image
    const preview = document.getElementById('imagePreview');
    preview.innerHTML = `<img src="${data.result_image}?t=${Date.now()}" alt="Detection result">`;
    
    // Display detection info
    const detectionInfo = document.getElementById('detectionInfo');
    detectionInfo.classList.remove('hidden');
    
    let html = `
        <div class="alert ${data.criminals_found > 0 ? 'alert-danger' : 'alert-success'}">
            <i class="fas ${data.criminals_found > 0 ? 'fa-exclamation-triangle' : 'fa-check-circle'}"></i>
            <div>
                <strong>Detection Complete</strong>
                <p>${data.total_faces} face(s) detected, ${data.criminals_found} criminal(s) found</p>
            </div>
        </div>
    `;
    
    data.detections.forEach(detection => {
        if (detection.type === 'criminal') {
            html += `
                <div class="detection-card">
                    <h4>
                        <i class="fas fa-exclamation-circle"></i>
                        ${detection.name}
                        <span class="badge badge-danger">CRIMINAL</span>
                    </h4>
                    <p><strong>Crime:</strong> ${detection.crime}</p>
                    <p><strong>Age:</strong> ${detection.age} | <strong>Gender:</strong> ${detection.gender}</p>
                    <p><strong>Location:</strong> ${detection.location}</p>
                    <p><strong>Confidence:</strong> ${detection.confidence}</p>
                </div>
            `;
        } else {
            html += `
                <div class="detection-card unknown">
                    <h4>
                        <i class="fas fa-user"></i>
                        ${detection.name}
                        <span class="badge badge-success">UNKNOWN</span>
                    </h4>
                    <p>Not in criminal database</p>
                </div>
            `;
        }
    });
    
    detectionInfo.innerHTML = html;
}

// Forms
function initForms() {
    // Add criminal form
    const addForm = document.getElementById('addCriminalForm');
    const criminalImage = document.getElementById('criminalImage');
    const fileName = document.getElementById('fileName');
    
    criminalImage.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            fileName.textContent = e.target.files[0].name;
        } else {
            fileName.textContent = 'No file chosen';
        }
    });
    
    addForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        e.stopPropagation();
        
        console.log('Form submission prevented, handling with JavaScript...');
        
        const formData = new FormData(addForm);
        
        console.log('Submitting add criminal form...');
        showLoading(true);
        
        try {
            const response = await fetch('/api/criminal/add', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            console.log('Add criminal response:', data);
            
            if (data.success) {
                showToast(data.message, 'success');
                addForm.reset();
                fileName.textContent = 'No file chosen';
                
                console.log('Reloading criminals list...');
                
                // Force reload criminals immediately
                setTimeout(async () => {
                    await loadCriminals();
                    console.log('Criminals reloaded after timeout');
                }, 100);
                
                // Switch to criminals tab to show the new addition
                setTimeout(() => {
                    switchSection('criminals');
                    console.log('Switched to criminals tab');
                }, 200);
                
            } else {
                console.error('Add criminal failed:', data.error);
                showToast('Error: ' + data.error, 'error');
            }
        } catch (error) {
            console.error('Add criminal error:', error);
            showToast('Error: ' + error.message, 'error');
        } finally {
            showLoading(false);
        }
        
        return false; // Prevent any default form submission
    });
    
    // Search
    const searchInput = document.getElementById('searchInput');
    let searchTimeout;
    
    searchInput.addEventListener('input', (e) => {
        clearTimeout(searchTimeout);
        searchTimeout = setTimeout(() => {
            searchCriminals(e.target.value);
        }, 500);
    });
    
    // Refresh stats
    document.getElementById('refreshStats').addEventListener('click', loadStatistics);
}

// Load criminals
async function loadCriminals() {
    try {
        console.log('Loading criminals...');
        const response = await fetch('/api/criminals');
        const data = await response.json();
        
        console.log('Criminals API response:', data);
        
        if (data.success) {
            console.log('Found', data.criminals.length, 'criminals');
            displayCriminals(data.criminals);
        } else {
            console.error('Failed to load criminals:', data.error);
            showToast('Failed to load criminals: ' + data.error, 'error');
        }
    } catch (error) {
        console.error('Error loading criminals:', error);
        showToast('Error loading criminals: ' + error.message, 'error');
    }
}

function displayCriminals(criminals) {
    console.log('Displaying criminals:', criminals);
    const container = document.getElementById('criminalsList');
    
    if (!container) {
        console.error('Criminals container not found!');
        return;
    }
    
    if (criminals.length === 0) {
        console.log('No criminals to display');
        container.innerHTML = `
            <div class="placeholder" style="grid-column: 1/-1; padding: 3rem;">
                <i class="fas fa-users"></i>
                <p>No criminals in database yet</p>
            </div>
        `;
        return;
    }
    
    console.log('Rendering', criminals.length, 'criminals');
    let html = '';
    criminals.forEach(criminal => {
        console.log('Rendering criminal:', criminal.name);
        html += `
            <div class="criminal-card">
                <h3><i class="fas fa-user-secret"></i> ${criminal.name}</h3>
                <p><strong>ID:</strong> #${criminal.id}</p>
                <p><strong>Crime:</strong> ${criminal.crime}</p>
                <p><strong>Age:</strong> ${criminal.age} | <strong>Gender:</strong> ${criminal.gender}</p>
                <p><strong>Location:</strong> ${criminal.location}</p>
                ${criminal.detection_count > 0 ? `
                    <span class="detection-badge">
                        <i class="fas fa-eye"></i> Detected ${criminal.detection_count} time(s)
                    </span>
                ` : ''}
                ${criminal.last_detected ? `
                    <p class="mt-1"><small>Last seen: ${criminal.last_detected}</small></p>
                ` : ''}
            </div>
        `;
    });
    
    container.innerHTML = html;
    console.log('Criminals displayed successfully');
}

// Search criminals
async function searchCriminals(query) {
    if (!query.trim()) {
        loadCriminals();
        return;
    }
    
    try {
        const response = await fetch(`/api/search?name=${encodeURIComponent(query)}`);
        const data = await response.json();
        
        if (data.success) {
            displayCriminals(data.results);
        }
    } catch (error) {
        console.error('Error searching:', error);
    }
}

// Load statistics
async function loadStatistics() {
    try {
        const response = await fetch('/api/statistics');
        const data = await response.json();
        
        if (data.success) {
            displayStatistics(data.statistics);
        }
    } catch (error) {
        console.error('Error loading statistics:', error);
    }
}

function displayStatistics(stats) {
    const container = document.getElementById('statsContainer');
    
    let html = `
        <div class="stats-overview">
            <div class="stat-card">
                <h3><i class="fas fa-database"></i> Total Criminals</h3>
                <div class="stat-value">${stats.total_criminals}</div>
                <div class="stat-label">In Database</div>
            </div>
            <div class="stat-card">
                <h3><i class="fas fa-eye"></i> Total Detections</h3>
                <div class="stat-value">${stats.total_detections}</div>
                <div class="stat-label">All Time</div>
            </div>
            <div class="stat-card">
                <h3><i class="fas fa-user-check"></i> Criminals Detected</h3>
                <div class="stat-value">${stats.criminals_detected}</div>
                <div class="stat-label">Ever Spotted</div>
            </div>
            <div class="stat-card">
                <h3><i class="fas fa-percentage"></i> Detection Rate</h3>
                <div class="stat-value">${stats.detection_rate.toFixed(1)}%</div>
                <div class="stat-label">Success Rate</div>
            </div>
        </div>
    `;
    
    if (stats.top_detected.length > 0) {
        html += `
            <div class="top-detected">
                <h3><i class="fas fa-trophy"></i> Most Detected Criminals</h3>
        `;
        
        stats.top_detected.forEach((criminal, index) => {
            html += `
                <div class="top-item">
                    <div class="top-item-info">
                        <h4>#${index + 1} ${criminal.name}</h4>
                        <p>${criminal.crime} | ${criminal.location}</p>
                    </div>
                    <div class="top-item-count">${criminal.detection_count}</div>
                </div>
            `;
        });
        
        html += `</div>`;
    }
    
    container.innerHTML = html;
}

// Utility functions
function showLoading(show) {
    const overlay = document.getElementById('loadingOverlay');
    if (show) {
        overlay.classList.remove('hidden');
    } else {
        overlay.classList.add('hidden');
    }
}

function showToast(message, type = 'success') {
    const toast = document.getElementById('toast');
    const toastMessage = document.getElementById('toastMessage');
    
    toastMessage.textContent = message;
    toast.classList.remove('hidden', 'error');
    
    if (type === 'error') {
        toast.classList.add('error');
    }
    
    setTimeout(() => {
        toast.classList.add('hidden');
    }, 5000);
}