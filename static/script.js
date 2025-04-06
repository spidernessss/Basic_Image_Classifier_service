// ASSSUMING THE SERVICE PORT IS 8010!
function showStep(stepId) {
    const steps = document.querySelectorAll('.step-container');
    steps.forEach(step => step.style.display = 'none');
    document.getElementById(stepId).style.display = 'block';
}
const formData = new FormData();

document.getElementById('load-dataset').addEventListener('click', () => {
    const datasetInput = document.getElementById('dataset-file');
    const datasetFile = datasetInput.files[0];

    if (!datasetFile) {
        alert('Please select a dataset file.');
        return;
    }

    formData.append('dataset', datasetFile);
    showStep('step2'); // Move to the next step
});

document.getElementById('load-image').addEventListener('click', () => {
    const imageInput = document.getElementById('image-file');
    const imageFile = imageInput.files[0];

    if (!imageFile) {
        alert('Please select an image file.');
        return;
    }

    formData.append('input_image', imageFile);
    fetch('/process', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            console.error(data.error);
            alert('Error loading dataset and image: ' + data.error + ' formdata ' + formData);
        } else {
            console.log('Dataset and Image loaded!');
            showStep('step3'); // Move to the next step
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while loading dataset and image.');
    });
});

document.getElementById('start-classification').addEventListener('click', () => {
    console.log('Classification started!');

    const filename = 'output_image.jpg';
    const outputImageSrc = 'http://localhost:8010/download/' + filename;

    const outputImage = document.getElementById('output-image');
    outputImage.src = outputImageSrc; // Set the image source to the FastAPI endpoint
    outputImage.style.display = 'block'; // Show the output image

    const downloadLink = document.getElementById('download-link');
    downloadLink.href = outputImageSrc; // Set the download link to the image source
    downloadLink.download = 'classified-image.jpg'; // Set the download filename
    downloadLink.innerText = 'Download Image'; // Set link text
    downloadLink.style.display = 'block'; // Show the download link

});

// Start Over
document.getElementById('step-4').addEventListener('click', () => {
    // Reset the UI to step 1
    showStep('step1');
});

// Initialize the UI (Show Step 1 by default)
showStep('step1');
