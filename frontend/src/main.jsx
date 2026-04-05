import ReactDOM from 'react-dom/client'
import App from './App'
import './App.css'

// NOTE: StrictMode removed intentionally.
// It double-invokes components/effects in dev mode, which caused
// duplicate API calls (upload firing twice → duplicate candidates).
ReactDOM.createRoot(document.getElementById('root')).render(
  <App />,
)
