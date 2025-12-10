import { Request, Response } from 'express';

export const renderHome = (req: Request, res: Response) => {
    res.render('home', { title: 'Home' });
};

export const renderChat = (req: Request, res: Response) => {
    res.render('chat', { title: 'Chat' });
};

export const renderAbout = (req: Request, res: Response) => {
    res.render('about', { title: 'About Us' });
};

export const renderProduct = (req: Request, res: Response) => {
    res.render('product', { title: 'Product Analysis' });
};
