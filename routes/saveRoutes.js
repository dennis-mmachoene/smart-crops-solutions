// routes/predictions.js
const express = require('express');
const router = express.Router();
const Prediction = require('../models/Prediction');

router.post('/save-prediction', async (req, res) => {
    try {
        // Extract data from the request body
        const {
            costCultivation,
            production,
            yield,
            temperature,
            rainFallAnnual,
            region,
            crop,
            regionalDemand,
            predictedPrice,
            priceTrend,
            recommendation,
            confidenceScore
        } = req.body;

        // Create a new prediction document
        const newPrediction = new Prediction({
            costCultivation,
            production,
            yield,
            temperature,
            rainFallAnnual,
            region,
            crop,
            regionalDemand,
            predictedPrice,
            priceTrend,
            recommendation,
            confidenceScore
        });

        // Save to MongoDB
        await newPrediction.save();

        // Respond with success
        res.status(201).json({ message: 'Prediction saved successfully' });
    } catch (error) {
        console.error('Error saving prediction:', error);
        res.status(500).json({ error: 'Failed to save prediction data' });
    }
});

module.exports = router;
