document.addEventListener('DOMContentLoaded', () => {
    const form = document.querySelector('form'); // Select the form

    form.addEventListener('submit', async (event) => {
        event.preventDefault(); // Prevent the default form submission

        const formData = new FormData(form); // Create a FormData object from the form

        try {
            const response = await fetch(form.action, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const result = await response.json(); // Assuming the backend returns JSON
            console.log('Success:', result);

            // Display the result image
            const resultImage = document.querySelector('img[alt="output"]');
            resultImage.src = `data:image/png;base64,${result.op}`; // Update the image source with the result

        } catch (error) {
            console.error('Error:', error);
        }
    });
});
