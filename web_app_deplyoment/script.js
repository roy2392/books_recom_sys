const form = document.getElementById('recommendation-form');
const recommendationsDiv = document.getElementById('recommendations');

form.addEventListener('submit', function (event) {
    event.preventDefault();
    const user_id = form.user_id.value;
    fetch('/recommendations', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded'
        },
        body: `user_id=${encodeURIComponent(user_id)}`
    })
    .then(response => response.json())
    .then(recommendations => {
        recommendationsDiv.innerHTML = '<h2>Recommended Books:</h2><ul>';
        recommendations.forEach(book => {
            recommendationsDiv.innerHTML += `<li>${book}</li>`;
        });
        recommendationsDiv.innerHTML += '</ul>';
    })
    .catch(error => {
        console.error('Error:', error);
    });
});
