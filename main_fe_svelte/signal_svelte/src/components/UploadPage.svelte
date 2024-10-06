<script>
    let files;
    let prediction = '';
    let url = '';

    async function handleSubmit(event) {
        event.preventDefault();
        const formData = new FormData();
        formData.append('file', files[0]);

        const response = await fetch('http://localhost:5000/upload', {
            method: 'POST',
            body: formData
        });

        console.log(response);

        if (response.ok) {
            const data = await response.json();
            prediction = data.prediction.precision;
            url = data.spectrogram_url;


        } else {
            console.error('Failed to upload file');
        }
    }
</script>

<style>
    .container {
        max-width: 600px;
        margin: 0 auto;
        padding: 2rem;
        text-align: center;
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }

    h1 {
        font-size: 2rem;
        margin-bottom: 1rem;
        color: #333;
    }

    form {
        margin-bottom: 1.5rem;
    }

    input[type="file"] {
        display: block;
        margin: 0 auto 1rem auto;
        padding: 0.5rem;
        border: 1px solid #ccc;
        border-radius: 4px;
    }

    button {
        padding: 0.5rem 1rem;
        font-size: 1rem;
        color: #fff;
        background-color: #007bff;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }

    button:hover {
        background-color: #0056b3;
    }

    .result {
        margin-top: 1.5rem;
        padding: 1rem;
        background-color: #e9ecef;
        border-radius: 4px;
    }

    .result h2 {
        margin-bottom: 0.5rem;
        color: #28a745;
    }

    .result a {
        color: #007bff;
        text-decoration: none;
    }

    .result a:hover {
        text-decoration: underline;
    }

    img {
        display: block;
        margin: 0 auto;
        max-width: 100%;
        border-radius: 4px;
    }
</style>

<div class="container">
    <h1>Upload a File</h1>
    <form on:submit={handleSubmit}>
        <input bind:files type="file" />
        <button type="submit">Upload</button>
    </form>

    {#if prediction && url}
        <div class="result">
            <h2>Prediction: {prediction}</h2>
            <img src={url} alt="Spectrogram" />
        </div>
    {/if}
</div>