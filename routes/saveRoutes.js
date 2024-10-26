const express = require('express');
const Prediction = require('../models/Prediction');
const router = express.Router();

router.post('/save-prediction', async (req, res) => {
    try {
        const { predictedPrice, priceTrend, recommendation, confidenceScore } = req.body;
        const prediction = new Prediction({
            predictedPrice,
            priceTrend,
            recommendation,
            confidenceScore,
        });
        await prediction.save();
        res.status(200).json({ message: 'Prediction saved successfully' });
    } catch (error) {
        console.error('Error saving prediction:', error);
        res.status(500).json({ message: 'Failed to save prediction' });
    }
})

router.get('/get-predictions', async (req, res) => {
    try {
        const predictions = await Prediction.find().sort({ createdAt: -1 });
        res.status(200).json(predictions);
    } catch (error) {
        console.error('Error fetching predictions:', error);
        res.status(500).json({ message: 'Failed to fetch predictions' });
    }
});

module.exports = router;
