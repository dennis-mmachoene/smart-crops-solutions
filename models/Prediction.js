// models/Prediction.js
const mongoose = require('mongoose');

const predictionSchema = new mongoose.Schema({
    costCultivation: Number,
    production: Number,
    yield: Number,
    temperature: Number,
    rainFallAnnual: Number,
    region: String,
    crop: String,
    regionalDemand: String,
    predictedPrice: Number,
    priceTrend: String,
    recommendation: String,
    confidenceScore: Number,
    createdAt: {
        type: Date,
        default: Date.now
    }
});

module.exports = mongoose.model('Prediction', predictionSchema);
