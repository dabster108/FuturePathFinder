// Helper: shuffle array
function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
}

// DOM elements
const form = document.getElementById('career-form');
const fieldSelect = document.getElementById('field_of_study');
const resultsSection = document.getElementById('results-section');
const recommendationsList = document.getElementById('recommendations-list');
const shuffleBtn = document.getElementById('shuffle-btn');
const saveBtn = document.getElementById('save-btn');

let currentRecommendations = [];

// Load fields of study
fetch('/fields')
    .then(res => res.json())
    .then(data => {
        data.fields.forEach(field => {
            const opt = document.createElement('option');
            opt.value = field;
            opt.textContent = field;
            fieldSelect.appendChild(opt);
        });
    });

// Handle form submission
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    resultsSection.classList.add('hidden');
    recommendationsList.innerHTML = '';
    const formData = new FormData(form);
    const payload = {
        field_of_study: formData.get('field_of_study'),
        university_gpa: parseFloat(formData.get('university_gpa')),
        internships_completed: parseInt(formData.get('internships_completed')),
        projects_completed: parseInt(formData.get('projects_completed')),
        certifications: parseInt(formData.get('certifications')),
        soft_skills_score: parseInt(formData.get('soft_skills_score'))
    };
    try {
        const res = await fetch('/recommend', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.recommendations) {
            currentRecommendations = data.recommendations;
            renderRecommendations(currentRecommendations);
            resultsSection.classList.remove('hidden');
        } else {
            recommendationsList.innerHTML = '<li>Sorry, no recommendations found.</li>';
            resultsSection.classList.remove('hidden');
        }
    } catch (err) {
        recommendationsList.innerHTML = '<li>Error fetching recommendations.</li>';
        resultsSection.classList.remove('hidden');
    }
});

// Render recommendations as horizontal cards with animation
function renderRecommendations(recs) {
    recommendationsList.innerHTML = '';
    recs.forEach((rec, idx) => {
        const li = document.createElement('li');
        li.className = 'recommendation-item';
        li.setAttribute('tabindex', '0');
        li.innerHTML = `
            <div style="flex:1;">
                <div class="career-title">${rec.career}
                    <span class="info-icon" tabindex="0" aria-label="Why this career was recommended" role="button">&#9432;
                        <span class="tooltip" role="tooltip">${rec.explanation}</span>
                    </span>
                </div>
                <div class="career-confidence">Confidence: ${rec.confidence.toFixed(1)}%</div>
                <div class="career-description">${rec.description}</div>
            </div>
        `;
        recommendationsList.appendChild(li);
    });
    animateCards();
}

// Animate cards sliding in from right using IntersectionObserver
function animateCards() {
    const cards = document.querySelectorAll('.recommendation-item');
    const observer = new window.IntersectionObserver((entries, obs) => {
        entries.forEach((entry, idx) => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                obs.unobserve(entry.target);
            }
        });
    }, { threshold: 0.2 });
    cards.forEach((card, i) => {
        card.classList.remove('visible');
        setTimeout(() => observer.observe(card), i * 120); // staggered
    });
}

// Shuffle button
shuffleBtn.addEventListener('click', () => {
    shuffleArray(currentRecommendations);
    renderRecommendations(currentRecommendations);
});

// Save button (download as JSON)
saveBtn.addEventListener('click', () => {
    const blob = new Blob([JSON.stringify(currentRecommendations, null, 2)], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'career_recommendations.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
});

// Keyboard accessibility for tooltips
// Show tooltip on focus, hide on blur
recommendationsList.addEventListener('focusin', (e) => {
    if (e.target.classList.contains('info-icon')) {
        const tip = e.target.querySelector('.tooltip');
        if (tip) {
            tip.style.visibility = 'visible';
            tip.style.opacity = '1';
        }
    }
});
recommendationsList.addEventListener('focusout', (e) => {
    if (e.target.classList.contains('info-icon')) {
        const tip = e.target.querySelector('.tooltip');
        if (tip) {
            tip.style.visibility = '';
            tip.style.opacity = '';
        }
    }
});