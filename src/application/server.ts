import express from 'express';
import path from 'path';
import bodyParser from 'body-parser';
import * as pageController from './controllers/pageController';
import * as apiController from './controllers/apiController';

const app = express();
const PORT = 3000;
// Middleware
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.set('view engine', 'ejs');
// Use process.cwd() to find views and public relative to the project root, regardless of where the script is run from
app.set('views', path.join(process.cwd(), 'views'));
app.use(express.static(path.join(process.cwd(), 'public')));

// Bootstrap static files
app.use('/css', express.static(path.join(process.cwd(), 'node_modules/bootstrap/dist/css')));
app.use('/js', express.static(path.join(process.cwd(), 'node_modules/bootstrap/dist/js')));

// Routes
app.get('/', pageController.renderHome);
app.get('/chat', pageController.renderChat);
app.get('/about', pageController.renderAbout);
app.get('/product', pageController.renderProduct);

app.post('/api/ask', apiController.askQuestion);
app.post('/api/product', apiController.getProductInfo);

app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
