import axios from 'axios';

const API_URL = 'http://localhost:5000';

export const predictYield = async (data) => {
  try {
    const response = await axios.post(`${API_URL}/predict`, data);
    return response.data;
  } catch (error) {
    if (error.response) {
      // The request was made and the server responded with a status code
      // that falls out of the range of 2xx
      throw new Error(error.response.data.message || 'Failed to fetch prediction from the model.');
    } else if (error.request) {
      // The request was made but no response was received
      throw new Error('No response from the prediction server. Is it running?');
    } else {
      // Something happened in setting up the request that triggered an Error
      throw new Error('An unexpected error occurred while processing your request.');
    }
  }
};
