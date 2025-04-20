// document.addEventListener('DOMContentLoaded', () => {
//     const form = document.querySelector('form'); // Select the form

//     form.addEventListener('submit', async (event) => {
//         event.preventDefault(); // Prevent the default form submission

//         const formData = new FormData(form); // Create a FormData object from the form

//         try {
//             console.log('Submitting form...'); // Debug log
//             const response = await fetch(form.action, {
//                 method: 'POST',
//                 body: formData,
//             });

//             if (!response.ok) {
//                 const errorText = await response.text(); // Get the error message
//                 console.error('Error:', errorText);
//                 throw new Error('Network response was not ok');
//             }

//             const result = await response.json(); // Assuming the backend returns JSON
//             console.log('Success:', result);

//             // Display the result image
//             const resultImage = document.querySelector('img[alt="output"]');
//             resultImage.src = `data:image/png;base64,${result.op}`; // Update the image source with the result

//         } catch (error) {
//             console.error('Error:', error);
//         }
//     });
// });


const modelInput = document.getElementById('modelImg');
const clothInput = document.getElementById('clothImg');
const submitBtn = document.getElementById('submitBtn');

const modelPreview = document.getElementById('modelPreview');
const clothPreview = document.getElementById('clothPreview');
const resultImg = document.getElementById('resultImg');

// For storing the selected file URLs
let modelImageURL = "";
let clothImageURL = "";

// Preview model image
modelInput.addEventListener('change', function () {
  const file = this.files[0];
  if (file) {
    modelImageURL = URL.createObjectURL(file);
    modelPreview.src = modelImageURL;
  }
});

// Preview clothing image
clothInput.addEventListener('change', function () {
  const file = this.files[0];
  if (file) {
    clothImageURL = URL.createObjectURL(file);
    clothPreview.src = clothImageURL;
  }
});

// Handle submit
submitBtn.addEventListener('click', function () {
  if (modelImageURL && clothImageURL) {
    // Simple "if same" logic – customize this for your needs
    if (modelImageURL === clothImageURL) {
      alert("You selected the same image for both model and clothing.");
      resultImg.src = modelImageURL;
    } else {
      // Simulating a try-on (replace this logic with Python backend call)
      resultImg.src = modelImageURL; // You can replace this with the processed result
    }
  } else {
    alert("Please choose both model and clothing images.");
  }
});
