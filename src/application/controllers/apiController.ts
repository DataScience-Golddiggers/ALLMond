import { Request, Response } from 'express';
import axios from 'axios';

const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://localhost:8000';

// Global Lock
let isProcessing = false;

export const askQuestion = async (req: Request, res: Response) => {
    if (isProcessing) {
        return res.status(429).json({ error: 'System is busy. Please try again later.' });
    }

    const { question } = req.body;
    if (!question) {
        return res.status(400).json({ error: 'Question is required' });
    }

    isProcessing = true;

    try {
        const response = await axios.post(`${AI_SERVICE_URL}/ask`, { question });
        res.json(response.data);
    } catch (error: any) {
        console.error('Error calling AI Service:', error.message);
        res.status(500).json({ error: 'Failed to get answer from AI service.' });
    } finally {
        isProcessing = false;
    }
};

export const getProductInfo = async (req: Request, res: Response) => {
    const { url } = req.body;
    
    if (!url) {
        return res.status(400).json({ error: 'URL is required' });
    }
    
    // Validate eBay URL
    const ebayRegex = /ebay\.(com|it)\/itm\//;
    if (!ebayRegex.test(url)) {
        return res.status(400).json({ error: 'Invalid eBay URL. Must contain ebay.com/itm/ or ebay.it/itm/' });
    }
    
    try {
        const response = await axios.post(`${AI_SERVICE_URL}/api/product`, { url });
        res.json(response.data);
    } catch (error: any) {
        console.error('Error calling AI Service for product:', error.message);
        res.status(500).json({ error: 'Failed to get product info.' });
    }
};
