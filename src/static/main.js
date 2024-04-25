/* Player and recorder setup */
const recorder = document.getElementById('recorder');
const player = document.getElementById('player');
const conversionForm = document.getElementById('conversion-form');
const submitButton = document.getElementById('submit');
const spinner = document.getElementById('spinner');
const summaryContainer = document.getElementById('textSummaryResult');

recorder.addEventListener('change', function (e) {
    const file = e.target.files[0];
    const url = URL.createObjectURL(file);
    player.src = url;
});

/* Form submission and processing */
conversionForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    submitButton.disabled = true;
    spinner.classList.remove('d-none');

    const formData = new FormData(conversionForm);
    const response = await fetch('/process_audio/', {
        method: 'POST',
        body: formData,
    });

    const data = await response.json();
    summaryContainer.innerHTML = `<p class="text-muted">Summary:</p><p>${data.summary}</p>`;

    submitButton.disabled = false;
    spinner.classList.add('d-none');
});
