import React from 'react';
import './Header.css';
import Logo from '../assets/cleverlearn-logos_white.png'

const Header = () => {
    return (
        <header>
            <div className="logo">
                <img src={Logo} alt="Logo" />
            </div>
            <nav>
                <ul>
                    <li><a href="https://cleverlearn.fr/#features" >L'outil CleverLearn</a></li>
                    <li><a href="https://cleverlearn.fr/#team" >Equipes et partenaires</a></li>
                    <li><a href="https://cleverlearn.fr/#faq" >Contact</a></li>
                </ul>
            </nav>
        </header>
    );
};

export default Header;