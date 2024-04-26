const fileInput = document.getElementById('file-input');
const submitButton = document.getElementById('submit-button');

submitButton.addEventListener('click', async () => {
  submitButton.disabled = true;
  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append('file', file);

  // Send the audio file to the server to process
  const response = await fetch('/process_audio/', {
    method: 'POST',
    body: formData
  });

  // Create a new Blob object for the summary text file returned from the server
  const blob = await response.blob();
  const downloadUrl = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = downloadUrl;
  a.download = 'summary.txt';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);

  submitButton.disabled = false;
});
