const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const path = require('path');
const connectDB = require('./config/db'); // Import the database connection
const authRoutes = require('./routes/authRoutes'); // Import authentication routes
const predictRoutes = require('./routes/predictRoutes');


const app = express();

// Middleware
app.use(cors());
app.use(bodyParser.json()); // for parsing application/json
app.use(bodyParser.urlencoded({ extended: true })); // for parsing application/x-www-form-urlencoded
app.use(express.static(path.join(__dirname, 'public'))); // Serve static files from the public directory

// Connect to MongoDB
connectDB();

// Route for GET request to serve the index.html
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'html', 'index.html'));
});
app.get('/homepage.html', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'html', 'homepage.html'))
});
// Route to serve the login.html
app.get('/login.html', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'html', 'login.html'));
});

// Use authentication routes
app.use('/auth', authRoutes);
app.use('/predict', predictRoutes);
// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});