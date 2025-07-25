// Helper: shuffle array (not used in current logic, but kept if you plan to use it)
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
const saveBtn = document.getElementById('save-btn');
const submitBtn = document.getElementById('submit-btn'); // Get a reference to the submit button for disabling

let currentRecommendations = [];

// Function to animate cards (vertical slide-in)
function animateVerticalCards() {
    const items = document.querySelectorAll('.recommendation-item');
    items.forEach((item, index) => {
        // Use a slight delay for a staggered effect
        item.style.animationDelay = `${index * 0.1}s`;
        item.classList.add('visible'); // Add a class to trigger CSS animation
    });
}

// Function to reset animation state
function resetAnimations() {
    const items = document.querySelectorAll('.recommendation-item');
    items.forEach(item => {
        item.classList.remove('visible');
        item.style.animationDelay = ''; // Clear delay for next animation
    });
}

// Load fields of study
async function loadFields() {
    try {
        const res = await fetch('/fields');
        if (!res.ok) {
            throw new Error(`HTTP error! status: ${res.status}`);
        }
        const data = await res.json();
        if (data.fields && data.fields.length > 0) {
            data.fields.forEach(field => {
                const opt = document.createElement('option');
                opt.value = field;
                opt.textContent = field;
                fieldSelect.appendChild(opt);
            });
            console.log('Fields of study loaded successfully.');
        } else {
            console.warn('No fields of study found in the response.');
            const opt = document.createElement('option');
            opt.value = "";
            opt.textContent = "Error: No fields available";
            fieldSelect.appendChild(opt);
            fieldSelect.disabled = true; // Disable if no fields loaded
        }
    } catch (err) {
        console.error('Error loading fields of study:', err);
        const opt = document.createElement('option');
        opt.value = "";
        opt.textContent = "Error loading fields";
        fieldSelect.appendChild(opt);
        fieldSelect.disabled = true; // Disable if error
    }
}

// Call loadFields when the script loads
document.addEventListener('DOMContentLoaded', loadFields);

// Handle form submission
form.addEventListener('submit', async (e) => {
    e.preventDefault(); // Prevent default form submission

    // Clear previous results and hide section immediately
    resultsSection.classList.add('hidden');
    recommendationsList.innerHTML = '';
    resetAnimations(); // Reset animation state before new recommendations

    // Disable submit button and add loading indicator
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<i data-lucide="loader" class="animate-spin"></i> <span>Processing...</span>';
    lucide.createIcons(); // Re-initialize icons for the new spinner

    const formData = new FormData(form);
    const payload = {
        field_of_study: formData.get('field_of_study'),
        university_gpa: parseFloat(formData.get('university_gpa')),
        internships_completed: parseInt(formData.get('internships_completed')),
        projects_completed: parseInt(formData.get('projects_completed')),
        certifications: parseInt(formData.get('certifications')),
        soft_skills_score: parseInt(formData.get('soft_skills_score'))
    };

    console.log('Sending payload:', payload); // Debugging: Check what's being sent

    try {
        const res = await fetch('/recommend', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!res.ok) { // Check if HTTP status is NOT 2xx
            const errorData = await res.json();
            throw new Error(errorData.error || `HTTP error! status: ${res.status}`);
        }

        const data = await res.json();
        console.log('Received data:', data); // Debugging: Check response data

        if (data.recommendations && data.recommendations.length > 0) {
            currentRecommendations = data.recommendations.slice(0, 3); // Take top 3
            renderRecommendations(currentRecommendations);
            resultsSection.classList.remove('hidden');
        } else {
            recommendationsList.innerHTML = '<li>Sorry, no recommendations found for your profile. Please adjust your input and try again.</li>';
            resultsSection.classList.remove('hidden');
            console.warn('API returned no recommendations or an empty list.');
        }
    } catch (err) {
        console.error('Error fetching recommendations:', err);
        recommendationsList.innerHTML = `<li>Error: ${err.message || 'Could not fetch recommendations. Please check your network connection and try again.'}</li>`;
        resultsSection.classList.remove('hidden');
    } finally {
        // Re-enable submit button and restore original text
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<i data-lucide="sparkles"></i> <span>Discover My Career Path</span>';
        lucide.createIcons(); // Re-initialize icons for the restored button
    }
});

// Render recommendations with modern animated cards
function renderRecommendations(recs) {
    recommendationsList.innerHTML = ''; // Clear existing list
    
    recs.forEach((rec, idx) => {
        const li = document.createElement('li');
        li.className = 'recommendation-item vertical-slide'; // Classes for styling and animation
        li.setAttribute('tabindex', '0'); // Make cards focusable for accessibility
        li.innerHTML = `
            <div style="flex:1;">
                <div class="career-title">${rec.career}
                    <span class="info-icon" tabindex="0" aria-label="Why this career was recommended" role="button">&#9432;
                        <span class="tooltip" role="tooltip">${rec.explanation}</span>
                    </span>
                </div>
                <div class="career-description">${rec.description}</div>
            </div>
        `;
        recommendationsList.appendChild(li);
    });
    animateVerticalCards(); // Trigger the animation after cards are added
}

// Save button (download as JSON)
saveBtn.addEventListener('click', () => {
    if (currentRecommendations.length === 0) {
        alert("No recommendations to save yet. Please generate some first!");
        return;
    }
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
    // Delegate event to check if the focused element is an info-icon
    if (e.target.classList.contains('info-icon')) {
        const tip = e.target.querySelector('.tooltip');
        if (tip) {
            tip.style.visibility = 'visible';
            tip.style.opacity = '1';
            // Optional: Adjust tooltip position if it goes off-screen (more advanced)
        }
    }
});
recommendationsList.addEventListener('focusout', (e) => {
    // Delegate event to check if the blurred element is an info-icon
    if (e.target.classList.contains('info-icon')) {
        const tip = e.target.querySelector('.tooltip');
        if (tip) {
            tip.style.visibility = ''; // Resets to CSS default
            tip.style.opacity = '';    // Resets to CSS default
        }
    }
});

// Sync range slider with number input
const softSkillsRange = document.getElementById('soft_skills_range');
const softSkillsScore = document.getElementById('soft_skills_score');

softSkillsRange.addEventListener('input', () => {
    softSkillsScore.value = softSkillsRange.value;
});

softSkillsScore.addEventListener('input', () => {
    softSkillsRange.value = softSkillsScore.value;
});