import React from 'react';
import { Link } from 'react-router-dom';

const Navbar: React.FC = () => {
  return (
    <nav className="bg-indigo-600 shadow-lg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex">
            <div className="flex-shrink-0 flex items-center">
              <Link to="/" className="text-white font-bold text-xl">ML Web App</Link>
            </div>
            <div className="hidden sm:ml-6 sm:flex sm:space-x-8">
              <Link
                to="/"
                className="text-white hover:text-indigo-100 px-3 py-2 rounded-md text-sm font-medium"
              >
                Home
              </Link>
              <Link
                to="/models"
                className="text-white hover:text-indigo-100 px-3 py-2 rounded-md text-sm font-medium"
              >
                Models
              </Link>
              <Link
                to="/upload"
                className="text-white hover:text-indigo-100 px-3 py-2 rounded-md text-sm font-medium"
              >
                Upload Model
              </Link>
            </div>
          </div>
        </div>
      </div>
      
      {/* Mobile menu */}
      <div className="sm:hidden">
        <div className="px-2 pt-2 pb-3 space-y-1">
          <Link
            to="/"
            className="text-white hover:bg-indigo-500 block px-3 py-2 rounded-md text-base font-medium"
          >
            Home
          </Link>
          <Link
            to="/models"
            className="text-white hover:bg-indigo-500 block px-3 py-2 rounded-md text-base font-medium"
          >
            Models
          </Link>
          <Link
            to="/upload"
            className="text-white hover:bg-indigo-500 block px-3 py-2 rounded-md text-base font-medium"
          >
            Upload Model
          </Link>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
