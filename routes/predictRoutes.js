// routes/predictRoutes.js
const express = require('express');
const router = express.Router();
const axios = require('axios');

const PYTHON_API_URL = 'http://localhost:5000/predict';

router.post('/predict', async (req, res) => {
    try {
        // Forward the request to Python API
        console.log(req.body)
        const response = await axios.post(PYTHON_API_URL, req.body);
        
        // Send the prediction results back to the client
        console.log(response.data)
        res.json(response.data);
    } catch (error) {
        console.error('Error:', error.response ? error.response.data : error.message);
        res.status(500).json({ 
            error: 'Failed to make a prediction',
            details: error.response ? error.response.data : error.message
        });
    }
});

module.exports = router;