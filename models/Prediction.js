const mongoose = require('mongoose');

const PredictionSchema = new mongoose.Schema({
    predictedPrice: Number,
    priceTrend: String,
    recommendation: String,
    confidenceScore: Number,
    createdAt: {
        type: Date,
        default: Date.now,
    },
});

module.exports = mongoose.model('Prediction', PredictionSchema);
